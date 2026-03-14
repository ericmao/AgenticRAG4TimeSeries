"""
Layer B / Control Plane contracts (MVP). Mirror schemas; validate outbound payloads.
TODO: Replace with package or git submodule.
"""
from .inference import (
    Hypothesis,
    InferenceRequest,
    InferenceResult,
    NormalizedEvent,
)

__all__ = [
    "Hypothesis",
    "InferenceRequest",
    "InferenceResult",
    "NormalizedEvent",
]
