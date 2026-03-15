"""
CERT → Layer C Episodes: load logon/device CSV (or minimal synthetic), group by user + fixed window, emit Episode JSONs.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, List

import pandas as pd

from src.contracts.episode import Episode

logger = logging.getLogger(__name__)

# Default burst tag threshold: total events in window >= this → add "burst"
DEFAULT_BURST_THRESHOLD = 50


def parse_ts_ms(x: Any) -> int:
    """
    Parse a value to epoch milliseconds (UTC).
    Supports: string datetime (interpreted as UTC), int/float (epoch ms or seconds if < 1e12).
    """
    if x is None:
        raise ValueError("parse_ts_ms: null value")
    if isinstance(x, (int, float)):
        v = int(x)
        if v < 0:
            raise ValueError(f"parse_ts_ms: negative numeric value {v}")
        # If looks like seconds (e.g. < 1e12), convert to ms
        if v < 1e12:
            v *= 1000
        return v
    if isinstance(x, str):
        s = x.strip()
        if not s:
            raise ValueError("parse_ts_ms: empty string")
        # Try numeric string
        try:
            return parse_ts_ms(float(s))
        except (ValueError, TypeError):
            pass
        # Parse as datetime, treat as UTC
        try:
            dt = pd.to_datetime(s, utc=True)
        except Exception:
            dt = pd.to_datetime(s)
            if dt.tzinfo is None:
                dt = dt.tz_localize("UTC", ambiguous="infer")
            else:
                dt = dt.tz_convert("UTC")
        return int(dt.timestamp() * 1000)
    if hasattr(x, "timestamp"):
        # datetime-like
        dt = x
        if getattr(dt, "tzinfo", None) is None:
            dt = pd.Timestamp(dt).tz_localize("UTC", ambiguous="infer")
        else:
            dt = pd.Timestamp(dt).tz_convert("UTC")
        return int(dt.timestamp() * 1000)
    raise ValueError(f"parse_ts_ms: unsupported type {type(x)}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_cert_data(data_dir: Path) -> pd.DataFrame:
    """
    Load CERT from data_dir: logon.csv + device.csv (columns: date, user, computer, activity).
    If either file is missing, return minimal synthetic DataFrame with at least 2 episodes worth of data.
    """
    logon_file = data_dir / "logon.csv"
    device_file = data_dir / "device.csv"
    if logon_file.exists() and device_file.exists():
        logon = pd.read_csv(logon_file)
        device = pd.read_csv(device_file)
        logon["source"] = "logon"
        device["source"] = "device"
        merged = pd.concat([logon, device], ignore_index=True)
        merged = merged.sort_values(["user", "date"])
        if "pc" in merged.columns and "computer" not in merged.columns:
            merged = merged.rename(columns={"pc": "computer"})
        return merged
    # Minimal synthetic: USER0001, USER0002, PC001-style; span 2+ windows so we get at least 2 episodes
    logger.info("CERT files not found; generating minimal synthetic data (at least 2 episodes).")
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    rows = []
    for u in ["USER0001", "USER0002"]:
        for day in range(0, 16):  # 16 days → 3 windows with window_days=7
            ts = base + timedelta(days=day, hours=10, minutes=0)
            rows.append({"date": ts, "user": u, "computer": "PC001", "activity": "logon", "source": "logon"})
            rows.append({"date": ts + timedelta(minutes=30), "user": u, "computer": "PC001", "activity": "logoff", "source": "logon"})
            if day % 2 == 0:
                rows.append({"date": ts + timedelta(minutes=15), "user": u, "computer": "PC002", "activity": "connect", "source": "device"})
    return pd.DataFrame(rows).sort_values(["user", "date"])


def _sequence_tags(activities: set[str], computers: set[str], total_events: int, burst_threshold: int) -> List[str]:
    """Base tags: logon, logoff, device. Heuristic: lateral (>=2 computers), burst (events >= burst_threshold)."""
    tags = []
    if "logon" in activities:
        tags.append("logon")
    if "logoff" in activities:
        tags.append("logoff")
    if "connect" in activities or "disconnect" in activities:
        tags.append("device")
    if len(computers) >= 2:
        tags.append("lateral")
    if total_events >= burst_threshold:
        tags.append("burst")
    return sorted(tags)


def _safe_filename(episode_id: str) -> str:
    """Return a safe filename (alphanumeric, hyphen, underscore only)."""
    return re.sub(r"[^\w\-]", "_", episode_id)


def build_episodes_from_cert(
    data_dir: str | Path,
    run_id: str = "cert-run-1",
    window_days: int = 7,
    burst_threshold: int = DEFAULT_BURST_THRESHOLD,
) -> List[Episode]:
    """
    Load CERT (or minimal synthetic), group by user + deterministic time window, return list of Episodes.
    Windowing: user_min_ts = min(ts_ms per user), window_index = floor((ts_ms - user_min_ts) / window_ms).
    """
    root = _repo_root()
    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        data_dir = root / data_dir

    merged = _load_cert_data(data_dir)
    if merged is None or merged.empty:
        return []

    merged["date"] = pd.to_datetime(merged["date"], utc=True)
    # Resolve ts_ms for every row (robust parse)
    ts_list = []
    for _, row in merged.iterrows():
        try:
            ts_list.append(parse_ts_ms(row["date"]))
        except Exception as e:
            logger.warning("Skipping row with invalid date %s: %s", row.get("date"), e)
            ts_list.append(None)
    merged["ts_ms"] = ts_list
    merged = merged.dropna(subset=["ts_ms"])
    merged["ts_ms"] = merged["ts_ms"].astype(int)

    window_ms = window_days * 24 * 3600 * 1000
    episodes: List[Episode] = []

    for user in merged["user"].unique():
        user_df = merged[merged["user"] == user].sort_values("ts_ms")
        if user_df.empty:
            continue
        user_min_ts = int(user_df["ts_ms"].min())
        user_df = user_df.copy()
        user_df["window_index"] = ((user_df["ts_ms"] - user_min_ts) // window_ms).astype(int)

        for win_idx in user_df["window_index"].unique():
            group = user_df[user_df["window_index"] == win_idx]
            if group.empty:
                continue
            t0_ms = user_min_ts + win_idx * window_ms
            t1_ms = t0_ms + window_ms
            # Only events strictly in [t0_ms, t1_ms)
            group = group[(group["ts_ms"] >= t0_ms) & (group["ts_ms"] < t1_ms)]
            if group.empty:
                continue

            events = []
            entities_set = set()
            computers_set = set()
            activities_set = set()
            for _, row in group.iterrows():
                ts_ms = int(row["ts_ms"])
                user_str = str(row["user"]).strip()
                computer_str = str(row["computer"]).strip()
                action = str(row["activity"]).strip().lower()
                source = str(row["source"]).strip().lower()
                if source not in ("logon", "device"):
                    source = "logon" if row["source"] == "logon" else "device"
                activities_set.add(action)
                computers_set.add(computer_str)
                entities_set.add(user_str)
                entities_set.add(computer_str)
                events.append({
                    "ts_ms": ts_ms,
                    "entity": user_str,
                    "action": action,
                    "artifact": {"host": computer_str},
                    "source": source,
                    "confidence": 0.8,
                    "domain": "IT",
                })
            events.sort(key=lambda e: e["ts_ms"])
            entities = sorted(entities_set)
            sequence_tags = _sequence_tags(activities_set, computers_set, len(events), burst_threshold)
            artifacts = [{"type": "user", "value": user}] + [{"type": "host", "value": c} for c in sorted(computers_set)]
            episode_id = f"cert-{user}-w{win_idx}"
            ep = Episode(
                episode_id=episode_id,
                run_id=run_id,
                t0_ms=t0_ms,
                t1_ms=t1_ms,
                entities=entities,
                artifacts=artifacts,
                sequence_tags=sequence_tags,
                events=events,
            )
            episodes.append(ep)

    return episodes


def write_episodes_to_dir(
    episodes: List[Episode],
    output_dir: str | Path,
) -> tuple[Path, int]:
    """
    Validate each Episode with Episode.model_validate, write to output_dir / {episode_id}.json.
    Skips and logs validation errors. Returns (output_dir, count_written).
    """
    root = _repo_root()
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for ep in episodes:
        try:
            # Validate by round-trip through model
            d = ep.model_dump()
            Episode.model_validate(d)
        except Exception as e:
            logger.warning("Skipping episode %s (validation error): %s", ep.episode_id, e)
            continue
        name = _safe_filename(ep.episode_id) + ".json"
        path = output_dir / name
        path.write_text(ep.model_dump_json(indent=2), encoding="utf-8")
        written += 1
    return output_dir, written


def run_cert_to_episodes(
    data_dir: str | Path = "data",
    out_dir: str | Path = "outputs/episodes/cert",
    window_days: int = 7,
    run_id: str = "cert-run-1",
    burst_threshold: int = DEFAULT_BURST_THRESHOLD,
) -> tuple[List[Episode], Path, int]:
    """Load CERT, build episodes, write to out_dir (validate each; skip on error). Returns (episodes, output_path, count_written)."""
    episodes = build_episodes_from_cert(
        data_dir, run_id=run_id, window_days=window_days, burst_threshold=burst_threshold
    )
    out_path, count_written = write_episodes_to_dir(episodes, out_dir)
    return episodes, out_path, count_written
