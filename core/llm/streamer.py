from __future__ import annotations

from collections.abc import Generator, Iterator


class TokenAggregator:
    """Remove reasoning blocks and incrementally form speakable phrases."""

    _OPEN_TAG = "<think>"
    _CLOSE_TAG = "</think>"
    _STRONG_PUNCTUATION = frozenset("。！？!?\n")
    _WEAK_PUNCTUATION = frozenset("，,；;：:")

    def __init__(self, min_chars: int = 8, max_chars: int = 42) -> None:
        if min_chars < 1 or max_chars < min_chars:
            raise ValueError("phrase lengths must satisfy 1 <= min_chars <= max_chars")
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.reset()

    def reset(self) -> None:
        self._filter_buffer = ""
        self._phrase_buffer = ""
        self._visible_parts: list[str] = []
        self._in_think = False

    @property
    def visible_text(self) -> str:
        return "".join(self._visible_parts).strip()

    @staticmethod
    def _partial_tag_length(text: str, tag: str) -> int:
        upper = min(len(text), len(tag) - 1)
        for size in range(upper, 0, -1):
            if text.endswith(tag[:size]):
                return size
        return 0

    def _filter_reasoning(self, token: str) -> str:
        self._filter_buffer += token
        visible: list[str] = []

        while self._filter_buffer:
            if self._in_think:
                end = self._filter_buffer.find(self._CLOSE_TAG)
                if end >= 0:
                    self._filter_buffer = self._filter_buffer[end + len(self._CLOSE_TAG) :]
                    self._in_think = False
                    continue
                keep = self._partial_tag_length(self._filter_buffer, self._CLOSE_TAG)
                self._filter_buffer = self._filter_buffer[-keep:] if keep else ""
                break

            start = self._filter_buffer.find(self._OPEN_TAG)
            if start >= 0:
                visible.append(self._filter_buffer[:start])
                self._filter_buffer = self._filter_buffer[start + len(self._OPEN_TAG) :]
                self._in_think = True
                continue

            keep = self._partial_tag_length(self._filter_buffer, self._OPEN_TAG)
            emit_to = len(self._filter_buffer) - keep
            visible.append(self._filter_buffer[:emit_to])
            self._filter_buffer = self._filter_buffer[emit_to:]
            break

        return "".join(visible)

    def _drain_phrases(self, text: str) -> list[str]:
        phrases: list[str] = []
        for char in text:
            self._phrase_buffer += char
            stripped_length = len(self._phrase_buffer.strip())
            strong_boundary = char in self._STRONG_PUNCTUATION
            if char == "." and len(self._phrase_buffer) >= 2:
                strong_boundary = not self._phrase_buffer[-2].isdigit()
            weak_boundary = char in self._WEAK_PUNCTUATION and stripped_length >= self.min_chars
            if strong_boundary or weak_boundary or stripped_length >= self.max_chars:
                phrase = self._phrase_buffer.strip()
                self._phrase_buffer = ""
                if phrase:
                    phrases.append(phrase)
        return phrases

    def feed(self, token: str) -> list[str]:
        if not token:
            return []
        visible = self._filter_reasoning(token)
        if not visible:
            return []
        self._visible_parts.append(visible)
        return self._drain_phrases(visible)

    def finish(self) -> list[str]:
        phrases: list[str] = []
        if not self._in_think and self._filter_buffer:
            visible = self._filter_buffer
            self._visible_parts.append(visible)
            phrases.extend(self._drain_phrases(visible))
        self._filter_buffer = ""
        phrase = self._phrase_buffer.strip()
        self._phrase_buffer = ""
        if phrase:
            phrases.append(phrase)
        return phrases

    def aggregate(self, token_stream: Iterator[str]) -> Generator[str, None, None]:
        self.reset()
        for token in token_stream:
            yield from self.feed(token)
        yield from self.finish()
