"""SenseNova TI2T/TI2TI rollout route built on the official serving loop."""

from __future__ import annotations

import asyncio
import math
import uuid
from typing import Any

from fastapi import HTTPException, Request

from lightllm.server.core.objs.sampling_params import SamplingParams
from lightllm.server.core.objs.x2i_params import X2IParams, X2IResponse

from .api_models import ChatCompletionRequest, ChatCompletionRequestV2, ImageConfig
from .api_openai import (
    _get_images_and_audios,
    _normalize_image_b64_for_multimodal,
    _raw_image_to_data_url,
)
from .build_prompt import build_prompt
from .httpserver.manager import HttpServerManager
from .multimodal_params import MultimodalParams
from .rl_models import RLRolloutRequest
from .trajectory_budget import require_server_limit, image_context_tokens as _image_context_tokens, span_budget, prompt_token_count


def _image_config(policy, seed: int) -> ImageConfig:
    values: dict[str, Any] = {
        "steps": policy.image_steps,
        "guidance_scale": 1.0,
        "image_guidance_scale": 1.0,
        "seed": seed,
        "num_images": 1,
        "cfg_norm": "none",
        "dynamic_resolution": False,
    }
    if policy.height is not None or policy.width is not None:
        values.update(height=policy.height, width=policy.width)
    elif policy.image_size is not None:
        values["image_size"] = policy.image_size
    return ImageConfig(**values)


def _sde_config(policy) -> dict[str, Any]:
    values = {
        "noise_level": policy.image_noise_level,
        "t_eps": policy.t_eps,
        "window_start": policy.sde_window_start,
        "window_end": policy.sde_window_end,
        "selected_steps": policy.sde_selected_steps,
        "indices": None if policy.sde_indices is None else tuple(policy.sde_indices),
    }
    return values


async def _one_rollout(
    request: RLRolloutRequest,
    seed: int,
    raw_request: Request,
    manager: HttpServerManager,
) -> dict[str, Any]:
    require_server_limit(request.max_sequence_length, manager.max_req_total_len)
    modalities = ["text"] if request.modality == "ti2t" else ["text", "image"]
    image_config = None if request.modality == "ti2t" else _image_config(request.image_policy, seed)
    chat_v2 = ChatCompletionRequestV2(
        messages=[message.model_dump() for message in request.messages],
        modalities=modalities,
        image_config=image_config,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        do_sample=True,
        seed=seed,
        max_tokens=request.max_sequence_length,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        stream=False,
        chat_template_kwargs={},
    )
    chat_request = ChatCompletionRequest(**chat_v2.model_dump())
    images, audios = _get_images_and_audios(chat_request)
    multimodal = MultimodalParams(images=images, audios=audios)
    input_image_num = len(multimodal.images)
    prompt = await build_prompt(chat_request, tools=None)

    image_start_tag = manager.tokenizer.image_start_tag
    image_start_id = manager.tokenizer.image_start_id
    image_tag = manager.tokenizer.image_tag
    expected_image_context_tokens = 0
    if image_config is not None:
        image_geometry = X2IParams()
        image_geometry.init_from_image_config(image_config)
        expected_image_context_tokens = _image_context_tokens(
            manager.tokenizer,
            width=int(image_geometry.width),
            height=int(image_geometry.height),
        )
    events: list[dict[str, Any]] = []
    generated_images = 0
    await multimodal.verify_and_preload(raw_request)
    probe_params = SamplingParams()
    probe_params.init(tokenizer=manager.tokenizer, max_new_tokens=1, add_special_tokens=False)
    initial_prompt_tokens = prompt_token_count(manager, prompt, multimodal, probe_params)
    if initial_prompt_tokens >= request.max_sequence_length:
        raise ValueError("RL input leaves no generation room under the total trajectory limit")
    context_tokens = initial_prompt_tokens
    image_context_tokens = 0
    text_only_tail = request.modality == "ti2t" or request.max_images == 0
    finish_reason = "length"

    while context_tokens < request.max_sequence_length:
        span = span_budget(request.max_sequence_length, context_tokens,
                           image_tokens=expected_image_context_tokens,
                           allow_image=request.modality == "ti2ti" and not text_only_tail)
        can_generate_image = span.allow_image
        text_only_tail = not can_generate_image
        span_sequence_limit = span.sequence_limit
        image_reserve = request.max_sequence_length - span_sequence_limit
        params = SamplingParams()
        kwargs: dict[str, Any] = {
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_new_tokens": span.max_tokens,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "input_penalty": False,
            "add_special_tokens": False,
            "seed": seed,
            "skip_special_tokens": False,
        }
        if not can_generate_image:
            kwargs["invalid_token_ids"] = [image_start_id]
        else:
            kwargs["stop_sequences"] = [image_start_tag]
        params.init(tokenizer=manager.tokenizer, **kwargs)
        params.verify()

        token_ids: list[int] = []
        logprobs: list[float] = []
        text_parts: list[str] = []
        span_finish_reason = None
        generator = manager.generate(
            prompt,
            params,
            multimodal.clone(),
            request=raw_request,
            max_req_total_len=span_sequence_limit,
        )
        span_prompt_tokens = 0
        async for _, text, metadata, finish_status in generator:
            token_id = int(metadata["id"])
            logprob = float(metadata["logprob"])
            if not math.isfinite(logprob):
                raise RuntimeError("non-finite selected-token logprob")
            token_ids.append(token_id)
            logprobs.append(logprob)
            if text != image_start_tag:
                text_parts.append(text)
            if finish_status.is_finished():
                span_finish_reason = finish_status.get_finish_reason()
                span_prompt_tokens = int(metadata.get("prompt_tokens", span_prompt_tokens))

        if not token_ids:
            raise RuntimeError("LightLLM returned an empty rollout span")
        if span_prompt_tokens < 1:
            raise RuntimeError("LightLLM omitted RL prompt-token usage")
        if span_prompt_tokens != context_tokens:
            raise RuntimeError("RL prompt re-encoding changed the trajectory token count")
        context_tokens = span_prompt_tokens + len(token_ids)
        output_text = "".join(text_parts)
        stopped_on_image = token_ids[-1] == image_start_id
        events.append(
            {
                "type": "text",
                "text": output_text,
                "token_ids": token_ids,
                "selected_token_logprobs": logprobs,
                "response_mask": [True] * len(token_ids),
                "stop_token": (
                    token_ids[-1]
                    if stopped_on_image or span_finish_reason not in {None, "length"}
                    else None
                ),
                "decoded_tokens": manager.tokenizer.decode(token_ids, skip_special_tokens=False),
            }
        )

        if not stopped_on_image:
            if span_finish_reason == "length" and image_reserve and context_tokens < request.max_sequence_length:
                # The image-aware span intentionally stopped early so a late
                # <img> action would still fit.  If no image was sampled, use
                # that reserved tail for text while masking further images.
                prompt += output_text
                text_only_tail = True
                continue
            finish_reason = span_finish_reason or ("length" if context_tokens == request.max_sequence_length else "stop")
            break
        if not can_generate_image:
            raise RuntimeError("LightLLM emitted a masked image action")

        prompt += output_text
        x2i_params = X2IParams()
        x2i_params.init_from_image_config(image_config)
        x2i_params.timestep_shift = request.image_policy.timestep_shift
        x2i_params.rl_config = _sde_config(request.image_policy)
        x2i_response = await manager.generate_image(
            prompt,
            x2i_params,
            multimodal.clone(),
            request=raw_request,
            input_image_num=input_image_num,
            return_response=True,
        )
        if not isinstance(x2i_response, X2IResponse) or not x2i_response.images:
            raise RuntimeError("LightX2V returned no image for an image action")
        actual_image_context_tokens = _image_context_tokens(
            manager.tokenizer,
            width=int(x2i_params.width),
            height=int(x2i_params.height),
        )
        if context_tokens + len(x2i_response.images) * actual_image_context_tokens > request.max_sequence_length:
            raise RuntimeError("generated image exceeded the RL max_sequence_length reserve")
        bundle_ids = x2i_response.trace_bundles or []
        if len(bundle_ids) != len(x2i_response.images):
            raise RuntimeError("LightX2V image and trace bundle counts differ")
        for image, bundle_id in zip(x2i_response.images, bundle_ids):
            encoded = image.decode("ascii") if isinstance(image, bytes) else image
            events.append(
                {
                    "type": "image",
                    "image": _raw_image_to_data_url(encoded, image_config.image_type),
                    "trace_bundle_key": bundle_id,
                    "sde_geometry": {
                        # generate_image may apply the official dynamic-resolution
                        # rule for image-conditioned requests. Return the actual
                        # X2V geometry, not merely the requested policy default.
                        "height": int(x2i_params.height),
                        "width": int(x2i_params.width),
                        "image_steps": request.image_policy.image_steps,
                        "timestep_shift": request.image_policy.timestep_shift,
                        "t_eps": request.image_policy.t_eps,
                        "noise_level": request.image_policy.image_noise_level,
                        "window_start": request.image_policy.sde_window_start,
                        "window_end": request.image_policy.sde_window_end,
                        "selected_steps": request.image_policy.sde_selected_steps,
                        "indices": request.image_policy.sde_indices,
                    },
                }
            )
            prompt += image_tag
            multimodal.add_image(
                {"type": "base64", "data": _normalize_image_b64_for_multimodal(encoded)}
            )
            generated_images += 1
            image_context_tokens += actual_image_context_tokens
            context_tokens += actual_image_context_tokens
        if generated_images >= request.max_images:
            text_only_tail = True
        if context_tokens == request.max_sequence_length:
            finish_reason = "length"
            break

    completion_tokens = sum(len(event["token_ids"]) for event in events if event["type"] == "text")
    if initial_prompt_tokens is None or context_tokens is None:
        raise RuntimeError("RL rollout produced no measurable sequence")
    sequence_tokens = initial_prompt_tokens + completion_tokens + image_context_tokens
    if sequence_tokens != context_tokens or sequence_tokens > request.max_sequence_length:
        raise RuntimeError("RL rollout violated its total sequence-length contract")
    return {
        "seed": seed,
        "finish_reason": finish_reason,
        "usage": {
            "prompt_tokens": initial_prompt_tokens,
            "completion_tokens": completion_tokens,
            "image_count": generated_images,
            "image_context_tokens": image_context_tokens,
            "sequence_tokens": sequence_tokens,
            "max_sequence_length": request.max_sequence_length,
        },
        "events": events,
    }


async def rl_rollouts(request: RLRolloutRequest, raw_request: Request, manager: HttpServerManager):
    if request.expected_policy_version != manager.rl_active_policy_version:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "stale policy version",
                "expected": request.expected_policy_version,
                "active": manager.rl_active_policy_version,
            },
        )
    # Submit all members of one RL group together so LightLLM's native
    # continuous scheduler can batch their prefill/decode work.
    rollouts = await asyncio.gather(
        *(_one_rollout(request, seed, raw_request, manager) for seed in request.seeds)
    )
    return {
        "id": f"rollout-{uuid.uuid4().hex}",
        "policy_version": manager.rl_active_policy_version,
        "modality": request.modality,
        "rollouts": rollouts,
    }
