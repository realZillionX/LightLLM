"""One total LM-position limit for ordinary and RL SenseNova trajectories."""

from dataclasses import dataclass


def prompt_token_count(manager, prompt: str, multimodal, sampling) -> int:
    """Count the exact NeoChat expansion without allocating image-cache entries."""
    tokenizer = manager.tokenizer
    if multimodal.audios:
        raise ValueError("SenseNova trajectories support text and images only")
    expanded = prompt.replace(tokenizer.image_tag,
                              tokenizer.image_start_tag + tokenizer.image_end_tag,
                              len(multimodal.images))
    count = len(tokenizer.encode(expanded, None, add_special_tokens=False))
    for image in multimodal.images:
        tokenizer.init_imageitem_extral_params(image, multimodal, sampling)
        count += tokenizer.get_image_token_length(image)
    return count


def require_server_limit(requested: int | None, server_limit: int) -> int:
    if requested is not None and requested != server_limit:
        raise ValueError(
            f"max_sequence_length={requested} must equal server max_req_total_len={server_limit}"
        )
    return server_limit


def image_context_tokens(tokenizer, *, width: int, height: int) -> int:
    grid = int(tokenizer.patch_size) * int(1 / float(tokenizer.downsample_ratio))
    if min(width, height) < grid or width % grid or height % grid:
        raise ValueError("image geometry must fit the model's token grid")
    # The sampled <img> is already a text action. Add the grid and </img>.
    return (width // grid) * (height // grid) + 1


@dataclass(frozen=True)
class SpanBudget:
    max_tokens: int
    sequence_limit: int
    allow_image: bool


def span_budget(limit: int, context: int, *, image_tokens: int = 0, allow_image: bool = False) -> SpanBudget:
    if context > limit:
        raise ValueError(f"trajectory uses {context} tokens, exceeding total limit {limit}")
    # Reserve image context plus one text position for continuation after it.
    reserve = image_tokens + 1 if allow_image else 0
    if limit - context <= reserve:
        allow_image, reserve = False, 0
    return SpanBudget(limit - context - reserve, limit - reserve, allow_image)
