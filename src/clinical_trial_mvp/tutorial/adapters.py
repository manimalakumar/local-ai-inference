"""Normalize engine transports; clinical workflow stays in the inherited agent."""

from __future__ import annotations

import json
import time
import urllib.request

from clinical_trial_mvp.serving.client import Generation, OpenAIClient, Sample, loopback_url


class OllamaAdapter:
    def __init__(self, base_url, model, generation=Generation(), *, context=8192, opener=None, clock=time.perf_counter):
        self.base_url = loopback_url(base_url).removesuffix("/v1")
        self.model, self.generation, self.context = model, generation, context
        # Reuse the same proxy/redirect policy as the common OpenAI transport.
        self.opener = opener or OpenAIClient(base_url, model).opener
        self.clock = clock

    def complete(self, messages, schema=None):
        cfg = self.generation
        payload = {"model": self.model, "messages": messages, "stream": True, "think": False,
                   "keep_alive": "10m", "options": {"temperature": cfg.temperature, "seed": cfg.seed,
                   "num_predict": cfg.max_tokens, "num_ctx": self.context, "top_p": 1.0,
                   "repeat_penalty": 1.0}}
        if schema is not None:
            payload["format"] = schema
        request = urllib.request.Request(self.base_url + "/api/chat", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
        sample, started, finished = Sample(), self.clock(), False
        try:
            with self.opener(request, timeout=cfg.timeout_seconds) as response:
                for raw in response:
                    if self.clock() - started > cfg.timeout_seconds:
                        raise TimeoutError("Streaming request exceeded its deadline")
                    event = json.loads(raw)
                    if event.get("error"):
                        raise ValueError(str(event["error"]))
                    text = (event.get("message") or {}).get("content", "")
                    if text:
                        if not isinstance(text, str):
                            raise ValueError("Expected text content")
                        if sample.ttft_seconds is None:
                            sample.ttft_seconds = self.clock() - started
                        sample.content += text
                    if event.get("done"):
                        finished = True
                        sample.finish_reason = event.get("done_reason")
                        for source, target in (("prompt_eval_count", "prompt_tokens"), ("eval_count", "output_tokens")):
                            value = event.get(source)
                            if value is not None and (type(value) is not int or value < 0):
                                raise ValueError("Invalid token usage")
                            setattr(sample, target, value)
                        break
            if not finished or not sample.content or sample.finish_reason != "stop":
                raise ValueError("Incomplete, truncated, or empty Ollama completion")
            sample.success = True
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            sample.error = f"{type(exc).__name__}: {exc}"
        sample.e2e_seconds = self.clock() - started
        decode = sample.e2e_seconds - (sample.ttft_seconds or 0)
        if sample.success and sample.output_tokens and sample.output_tokens > 1 and decode > 0:
            sample.decode_tokens_per_second = (sample.output_tokens - 1) / decode
        return sample


class NIMAdapter(OpenAIClient):
    """NIM uses standard fields; profile-specific extensions are never sent."""

    def __init__(self, base_url, model, generation=Generation(), *, structured_outputs=False, **kwargs):
        super().__init__(base_url, model, generation, request_profile="nim", **kwargs)
        self.structured_outputs = structured_outputs

    def complete(self, messages, schema=None):
        return super().complete(messages, schema if self.structured_outputs else None)


def serving_adapter(runtime, base_url, model, generation, *, context=8192, structured_outputs=False):
    if runtime == "ollama":
        return OllamaAdapter(base_url, model, generation, context=context)
    if runtime == "nim":
        return NIMAdapter(base_url, model, generation, structured_outputs=structured_outputs)
    if runtime in {"llama_cpp", "vllm", "sglang"}:
        return OpenAIClient(base_url, model, generation, request_profile=runtime)
    raise ValueError("Unsupported tutorial serving adapter")
