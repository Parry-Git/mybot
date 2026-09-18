from core.llm.streamer import TokenAggregator


def test_filters_split_think_tags_and_preserves_visible_text() -> None:
    aggregator = TokenAggregator(min_chars=5, max_chars=30)
    phrases = list(
        aggregator.aggregate(iter(["<thi", "nk>不要朗读", "</thi", "nk>你好", "，这是测试", "。"]))
    )

    assert phrases == ["你好，这是测试。"]
    assert aggregator.visible_text == "你好，这是测试。"


def test_emits_strong_punctuation_without_waiting_for_minimum() -> None:
    aggregator = TokenAggregator(min_chars=8, max_chars=20)

    assert aggregator.feed("好。下一句") == ["好。"]
    assert aggregator.finish() == ["下一句"]


def test_forces_a_boundary_for_unpunctuated_text() -> None:
    aggregator = TokenAggregator(min_chars=4, max_chars=6)

    assert aggregator.feed("一二三四五六七") == ["一二三四五六"]
    assert aggregator.finish() == ["七"]


def test_unfinished_tag_at_max_length_does_not_drop_text() -> None:
    aggregator = TokenAggregator(min_chars=2, max_chars=6)
    emitted = aggregator.feed("12345<thi") + aggregator.finish()
    assert "".join(emitted) == "12345<thi"
    assert aggregator.visible_text == "12345<thi"
