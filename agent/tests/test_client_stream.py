from conftest import delta, tc

from frank_local.client import StreamAccumulator


def test_text_and_reasoning():
    acc = StreamAccumulator()
    ev = acc.feed(delta(reasoning="think "))
    ev += acc.feed(delta(content="Hello"))
    ev += acc.feed(delta(content=" world"))
    assert acc.text == "Hello world" and acc.reasoning == "think "
    assert ev == [("reasoning", "think "), ("text", "Hello"), ("text", " world")]
    assert acc.tool_calls() == []


def test_tool_call_split_across_chunks():
    acc = StreamAccumulator()
    acc.feed(delta(tool_calls=[tc(0, id="call_1", name="bash", arguments="")]))
    acc.feed(delta(tool_calls=[tc(0, arguments='{"command": ')]))
    acc.feed(delta(tool_calls=[tc(0, arguments='"ls"}')]))
    acc.feed(delta(tool_calls=[tc(1, id="call_2", name="read_file", arguments='{"path":"x"}')]))
    calls = acc.tool_calls()
    assert [c.name for c in calls] == ["bash", "read_file"]
    assert calls[0].id == "call_1" and calls[0].parsed() == {"command": "ls"}
    assert calls[1].parsed() == {"path": "x"}


def test_missing_id_is_synthesised():
    acc = StreamAccumulator()
    acc.feed(delta(tool_calls=[tc(0, name="bash", arguments="{}")]))
    (call,) = acc.tool_calls()
    assert call.id.startswith("call_") and call.parsed() == {}
