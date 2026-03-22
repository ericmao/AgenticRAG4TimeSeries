"""Thin wrapper around pipeline writeback (dry_run default); no brokers."""
from __future__ import annotations

from pathlib import Path
from typing import Union

from src.contracts.writeback import WritebackPatch
from src.pipeline.writeback_pipeline import run_writeback


def run_writeback_dry_run(episode_path: Union[str, Path], mode: str = "dry_run") -> tuple[WritebackPatch, Path]:
    """Load episode + on-disk agent outputs; build patch and write outputs/writeback + audit."""
    return run_writeback(episode_path, mode=mode)
