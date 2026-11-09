"""Canonicalize bare CUDA UUIDs for nvidia-smi without guessing a GPU index."""
import re


def gpu_uuid(value):
    if value is None:
        return None
    text = str(value)
    if re.fullmatch(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', text):
        return 'GPU-' + text
    return text or None
