"""NIM profile discovery and deployment plans; no assumed RTX compatibility."""

import re


def runnable_profiles(output):
    """Only accept IDs in the explicitly runnable block, never low-memory/incompatible blocks."""
    active, ids, heading_indent = False, [], 0
    for line in output.splitlines():
        if "Compatible with system and runnable:" in line:
            active = True
            heading_indent = len(line) - len(line.lstrip())
            continue
        if (active and line.strip().endswith(":") and line.lstrip().startswith("-")
                and len(line) - len(line.lstrip()) <= heading_indent):
            active = False
        if "Compatible with system but" in line or "Incompatible with system:" in line:
            active = False
        if active:
            match = re.search(r"\b[0-9a-f]{64}\b", line)
            if match:
                ids.append(match.group())
    return ids


def nim_gate(profile, *, model, hardware, enabled, system):
    if not enabled:
        return False, "NIM disabled; deployment/profile compatibility is unmeasured."
    if system != "Linux":
        return False, "NIM deployment requires the Linux GPU host."
    if not profile:
        return False, "No profile discovery evidence supplied; compatibility is unknown."
    if profile.get("model") != model or profile.get("hardware") != hardware:
        return False, "Profile evidence is for a different model or GPU."
    if not re.fullmatch(r"nvcr\.io/[A-Za-z0-9_./-]+@sha256:[0-9a-f]{64}", profile.get("image", "")):
        return False, "Pin the NIM image by digest before deploying."
    selected = profile.get("profile_id")
    if selected not in runnable_profiles(profile.get("discovery_output", "")):
        return False, "Selected profile is absent from the compatible-and-runnable list; skipped."
    return True, "Profile is in recorded runnable output for this model/GPU; confirm startup readiness."


def nim_commands(profile, *, port=8000):
    if not 1 <= port <= 65535:
        raise ValueError("Invalid NIM port")
    image = profile["image"]
    # Pass credentials by environment name only; never serialize their values.
    common = ["docker", "run", "--rm", "--gpus=all", "-e", "NGC_API_KEY", "-e", "HF_TOKEN"]
    return {"discover": [*common, image, "list-model-profiles"],
            "deploy": [*common, "--name", "clinical-trial-nim", "--shm-size=16g",
                       "-p", f"127.0.0.1:{port}:8000", "-e", "NIM_MODEL_PROFILE=" + profile["profile_id"],
                       image]}
