# Third-party dependencies and publication scope

This source tree integrates with installed Python packages and separately launched engines. It does not vendor engine implementations or distribute model weights.

| Component | Upstream license | Reference |
| --- | --- | --- |
| vLLM | Apache License 2.0 | https://github.com/vllm-project/vllm/blob/main/LICENSE |
| SGLang | Apache License 2.0 | https://github.com/sgl-project/sglang/blob/main/LICENSE |

The engine licenses are independent of this project. Installing or calling these engines does not assign their license to the surrounding project. Retain applicable notices if separately redistributing upstream code or distributions. Other Python dependencies remain governed by their upstream licenses; this file is not a complete transitive dependency bill of materials.

Model weights have separate repository-specific licenses and terms. Download them explicitly from their upstream repositories, retain the relevant model notices, and check the terms before redistributing weights. Model files, embedding checkpoints, and vendored native engines are excluded from the Git export.

**Project license:** no project-level license has been selected or found. The export deliberately does not invent a copyright owner or grant a new license. This does not prevent a Git commit. Choose an appropriate project license before offering general public reuse. Engine license information alone does not establish the rights to publish the surrounding project.
