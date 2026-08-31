from __future__ import annotations

import unittest

from lightllm.server.embed_cache.manager import _serve_after_listening


class _Pipe:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def send(self, message: str) -> None:
        self.events.append(f"send:{message}")


class _Server:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.active = True

    def _listen(self) -> None:
        self.events.append("listen")

    def _register(self) -> None:
        self.events.append("register")

    def accept(self) -> None:
        self.events.append("accept")
        self.active = False

    def close(self) -> None:
        self.events.append("close")


class EmbedCacheReadinessTest(unittest.TestCase):
    def test_readiness_follows_listener_registration(self) -> None:
        events: list[str] = []

        _serve_after_listening(_Server(events), _Pipe(events))

        self.assertEqual(
            events,
            ["listen", "register", "send:init ok", "accept", "close"],
        )


if __name__ == "__main__":
    unittest.main()
