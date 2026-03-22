"""file_llm_prompt：單檔 summary JSON 解析。"""
from __future__ import annotations

from src.kb.file_llm_prompt import parse_file_summary_json


def test_parse_file_summary_plain() -> None:
    text = '{"summary": "這是概要"}'
    assert parse_file_summary_json(text) == "這是概要"


def test_parse_file_summary_fenced() -> None:
    text = '```json\n{"summary": "x"}\n```'
    assert parse_file_summary_json(text) == "x"


def test_parse_file_summary_invalid() -> None:
    assert parse_file_summary_json("no json") == ""
