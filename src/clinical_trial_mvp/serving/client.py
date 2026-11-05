"""Shared OpenAI-compatible streaming transport for vLLM and SGLang."""

from __future__ import annotations

import json
import math
import time
import urllib.request
from dataclasses import asdict, dataclass
from typing import Callable
from urllib.parse import urlsplit


def loopback_url(value: str) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path.rstrip("/") not in {"", "/v1"}):
        raise ValueError("Use an HTTP loopback endpoint, optionally ending in /v1; use SSH forwarding for a GPU host.")
    _ = parsed.port  # Reject malformed ports before making any requests.
    return value.rstrip("/").removesuffix("/v1") + "/v1"


@dataclass(frozen=True)
class Generation:
    max_tokens: int = 1024
    temperature: float = 0.0
    seed: int = 42
    timeout_seconds: float = 180.0

    def __post_init__(self):
        if self.max_tokens < 1 or not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("Invalid generation token cap or temperature")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")


@dataclass
class Sample:
    success: bool = False
    content: str = ""
    error: str | None = None
    finish_reason: str | None = None
    ttft_seconds: float | None = None
    e2e_seconds: float = 0.0
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    decode_tokens_per_second: float | None = None

    def to_dict(self):
        return asdict(self)


def sse_events(response):
    """Parse framed SSE, including comments, CRLF, and multi-line data fields."""
    data = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    if data:
        yield "\n".join(data)


class OpenAIClient:
    def __init__(self, base_url: str, model: str, generation: Generation = Generation(), *,
                 opener: Callable | None = None, clock: Callable = time.perf_counter,
                 request_profile: str = "vllm", enable_thinking: bool = False):
        self.base_url = loopback_url(base_url)
        if not model.strip():
            raise ValueError("model must be nonempty")
        self.model, self.generation = model, generation
        if request_profile not in {"vllm", "sglang", "llama_cpp", "nim"}:
            raise ValueError("Unknown serving request profile")
        self.request_profile = request_profile
        self.enable_thinking = enable_thinking
        # Disable environment proxies and redirects: local requests stay local.
        self.opener = opener or urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect()
        ).open
        self.clock = clock

    def complete(self, messages: list[dict], schema: dict | None = None) -> Sample:
        cfg = self.generation
        payload = {"model": self.model, "messages": messages, "stream": True,
                   "stream_options": {"include_usage": True}, "max_tokens": cfg.max_tokens,
                   "temperature": cfg.temperature, "seed": cfg.seed, "top_p": 1.0,
                   "frequency_penalty": 0.0, "presence_penalty": 0.0}
        if self.request_profile in {"vllm", "sglang"}:
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        if schema is not None:
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "clinical_trial", "strict": True, "schema": schema}}
        request = urllib.request.Request(self.base_url + "/chat/completions",
                                         data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
        sample = Sample()
        started = self.clock()
        done = False
        try:
            with self.opener(request, timeout=cfg.timeout_seconds) as response:
                for body in sse_events(response):
                    if self.clock() - started > cfg.timeout_seconds:
                        raise TimeoutError("Streaming request exceeded its deadline")
                    if body == "[DONE]":
                        done = True
                        break
                    event = json.loads(body)
                    if event.get("error"):
                        raise ValueError(str(event["error"]))
                    usage = event.get("usage") or {}
                    for source, target in (("prompt_tokens", "prompt_tokens"),
                                           ("completion_tokens", "output_tokens")):
                        count = usage.get(source)
                        if count is not None:
                            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                                raise ValueError("Invalid token usage")
                            setattr(sample, target, count)
                    for choice in event.get("choices", []):
                        if choice.get("index", 0) != 0:
                            continue
                        content = (choice.get("delta") or {}).get("content")
                        if content:
                            if not isinstance(content, str):
                                raise ValueError("Expected a text content delta")
                            if sample.ttft_seconds is None:
                                sample.ttft_seconds = self.clock() - started
                            sample.content += content
                        if choice.get("finish_reason") is not None:
                            sample.finish_reason = choice["finish_reason"]
            if not done or sample.finish_reason is None or not sample.content:
                raise ValueError("Incomplete or empty completion stream")
            if sample.finish_reason != "stop":
                raise ValueError(f"Completion did not finish normally: {sample.finish_reason}")
            sample.success = True
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            sample.error = f"{type(exc).__name__}: {exc}"
        sample.e2e_seconds = self.clock() - started
        # SSE chunks are not tokens; use server usage only. This is a client estimate.
        decode_seconds = sample.e2e_seconds - (sample.ttft_seconds or 0)
        if sample.success and sample.output_tokens and sample.output_tokens > 1 and decode_seconds > 0:
            sample.decode_tokens_per_second = (sample.output_tokens - 1) / decode_seconds
        return sample


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Serving endpoints must not redirect")
