"""Offline wire-contract and timing tests: no server, engine, or GPU imports."""

import io
import json
from urllib.error import HTTPError

import pytest

from clinical_trial_mvp.serving.client import Generation, OpenAIClient, loopback_url, sse_events


def wire(events):
    return io.BytesIO(b"".join(
        b"data: " + (json.dumps(e).encode() if not isinstance(e, str) else e.encode()) + b"\r\n\r\n"
        for e in events))


def delta(content=None, finish=None):
    return {"choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish}]}


def invoke(events, **kwargs):
    return OpenAIClient("http://127.0.0.1:8000", "test", opener=lambda *a, **k: wire(events), **kwargs).complete([])


def test_sse_comments_crlf_multiline_and_trailing_event():
    raw = io.BytesIO(b': keepalive\r\nevent: message\r\ndata: {"a":\r\ndata: 1}\r\n\r\ndata: [DONE]')
    assert list(sse_events(raw)) == ['{"a":\n1}', '[DONE]']


def test_stream_contract_and_first_content_timing():
    seen = {}
    events = [delta(), delta("hello"), delta(" world"), delta(finish="stop"),
              {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 3}}, "[DONE]"]
    def opener(request, timeout):
        seen.update(url=request.full_url, payload=json.loads(request.data), timeout=timeout)
        return wire(events)
    ticks = iter([0, .1, .2, .25, .3, .4, .5, .6, 1.0])
    client = OpenAIClient("http://localhost:8000/v1/", "same-model", opener=opener, clock=lambda: next(ticks))
    result = client.complete([{"role": "user", "content": "hello"}], {"type": "object"})
    assert result.success and result.content == "hello world"
    assert result.ttft_seconds == .25 and result.e2e_seconds == 1.0
    assert result.prompt_tokens == 20 and result.output_tokens == 3
    assert result.decode_tokens_per_second == pytest.approx(2 / .75)
    assert seen["url"] == "http://localhost:8000/v1/chat/completions"
    payload = seen["payload"]
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["temperature"] == 0 and payload["seed"] == 42 and payload["top_p"] == 1


@pytest.mark.parametrize("events,fragment", [
    ([delta("partial")], "Incomplete"),
    ([delta("x"), "[DONE]"], "Incomplete"),
    ([delta(finish="stop"), "[DONE]"], "empty"),
    ([delta("truncated", "length"), "[DONE]"], "length"),
    ([delta("x", "content_filter"), "[DONE]"], "content_filter"),
    ([{"error": {"message": "OOM"}}], "OOM"),
    (["not json"], "JSONDecodeError"),
    ([{"usage": {"completion_tokens": -1}}], "Invalid token usage"),
    ([{"usage": {"completion_tokens": True}}], "Invalid token usage"),
])
def test_stream_failure_is_never_success(events, fragment):
    result = invoke(events)
    assert not result.success and fragment in result.error


def test_missing_usage_stays_unknown_and_never_counts_chunks_as_tokens():
    result = invoke([delta("one"), delta("two", "stop"), "[DONE]"])
    assert result.success
    assert result.output_tokens is None and result.decode_tokens_per_second is None


@pytest.mark.parametrize("exc", [TimeoutError("timeout"), OSError("reset"),
                                 HTTPError("http://localhost", 503, "unavailable", {}, None)])
def test_transport_errors_are_recorded(exc):
    def fail(*args, **kwargs):
        raise exc
    result = OpenAIClient("http://localhost", "model", opener=fail).complete([])
    assert not result.success and result.error


def test_stream_deadline():
    ticks = iter([0, 2, 2])
    result = invoke([delta("late")], generation=Generation(timeout_seconds=1), clock=lambda: next(ticks))
    assert not result.success and "deadline" in result.error


@pytest.mark.parametrize("url", ["https://localhost", "http://remote", "http://localhost.evil",
                                 "http://user:secret@localhost", "http://localhost/other",
                                 "http://localhost?secret=x", "http://localhost#x", "http://localhost:bad"])
def test_invalid_endpoints_fail_before_network(url):
    with pytest.raises(ValueError):
        loopback_url(url)


@pytest.mark.parametrize("kwargs", [{"max_tokens": 0}, {"temperature": float("nan")},
                                     {"timeout_seconds": 0}, {"timeout_seconds": float("inf")}])
def test_invalid_generation(kwargs):
    with pytest.raises(ValueError):
        Generation(**kwargs)
