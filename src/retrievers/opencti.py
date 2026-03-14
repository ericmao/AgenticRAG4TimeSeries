"""
OpenCTI retriever stub. Returns [] if OPENCTI_URL/TOKEN missing or RUN_MODE=dry_run. No network in dry_run.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from src.contracts.evidence import EvidenceItem

if TYPE_CHECKING:
    from src.contracts.episode import Episode, Hypothesis


def retrieve_from_opencti(
    episode: "Episode",
    hypothesis: Optional["Hypothesis"],
    queries: list[str],
) -> List[EvidenceItem]:
    """
    Placeholder: return [] if OpenCTI not configured or RUN_MODE=dry_run.
    No network call in dry_run.
    """
    from src.config import get_config
    cfg = get_config()
    if cfg.RUN_MODE == "dry_run":
        return []
    if not cfg.OPENCTI_URL or not cfg.OPENCTI_TOKEN:
        return []
    # Stub: no actual OpenCTI API call yet
    return []
