# Third-party dependencies and publication scope

This source tree integrates with installed Python packages and separately launched engines. It does not vendor engine implementations or distribute model weights.

| Component | Upstream license | Reference |
| --- | --- | --- |
| vLLM | Apache License 2.0 | https://github.com/vllm-project/vllm/blob/main/LICENSE |
| SGLang | Apache License 2.0 | https://github.com/sgl-project/sglang/blob/main/LICENSE |

The engine licenses are independent of this project. Installing or calling these engines does not assign their license to the surrounding project. Retain applicable notices if separately redistributing upstream code or distributions.

Model weights have separate repository-specific licenses and terms. Download them explicitly from their upstream repositories.

