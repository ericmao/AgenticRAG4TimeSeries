"""group_llm_prompt：JSON 解析。"""
from __future__ import annotations

from src.kb.group_llm_prompt import parse_llm_group_json


def test_parse_llm_group_json_plain() -> None:
    text = '{"meaning": "A", "usage_direction": "B", "threats": "C"}'
    d = parse_llm_group_json(text)
    assert d["meaning"] == "A"
    assert d["usage_direction"] == "B"
    assert d["threats"] == "C"


def test_parse_llm_group_json_fenced() -> None:
    text = """Here:
```json
{"meaning": "x", "usage_direction": "y", "threats": "z"}
```
"""
    d = parse_llm_group_json(text)
    assert d["meaning"] == "x"


def test_parse_llm_group_json_invalid() -> None:
    d = parse_llm_group_json("not json")
    assert d["meaning"] == ""
