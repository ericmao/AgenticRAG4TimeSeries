"""
Tests for CERT → Episode pipeline (cert_to_episodes).
Verifies r3.2-style data (pc column) is normalized to computer for downstream use.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


def test_load_cert_data_renames_pc_to_computer():
    """When logon/device CSVs use 'pc' (CERT r3.2), _load_cert_data renames to 'computer'."""
    from src.pipeline.cert_to_episodes import _load_cert_data

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        logon_csv = data_dir / "logon.csv"
        device_csv = data_dir / "device.csv"
        logon_csv.write_text(
            "id,date,user,pc,activity\n"
            "id1,01/01/2010 06:20:00,USER1,PC-001,Logon\n"
            "id2,01/01/2010 06:30:00,USER1,PC-001,Logoff\n",
            encoding="utf-8",
        )
        device_csv.write_text(
            "id,date,user,pc,activity\n"
            "id1,01/01/2010 07:00:00,USER1,PC-001,Connect\n",
            encoding="utf-8",
        )
        merged = _load_cert_data(data_dir)
    assert merged is not None and len(merged) > 0
    assert "computer" in merged.columns
    assert "pc" not in merged.columns
    assert merged["computer"].iloc[0] == "PC-001"


def test_load_cert_data_preserves_computer_column():
    """When CSVs already have 'computer', no rename and column is preserved."""
    from src.pipeline.cert_to_episodes import _load_cert_data

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        logon_csv = data_dir / "logon.csv"
        device_csv = data_dir / "device.csv"
        logon_csv.write_text(
            "date,user,computer,activity\n"
            "2020-01-01 06:20:00,USER1,PC001,logon\n",
            encoding="utf-8",
        )
        device_csv.write_text(
            "date,user,computer,activity\n"
            "2020-01-01 07:00:00,USER1,PC001,connect\n",
            encoding="utf-8",
        )
        merged = _load_cert_data(data_dir)
    assert "computer" in merged.columns
    assert merged["computer"].iloc[0] == "PC001"
