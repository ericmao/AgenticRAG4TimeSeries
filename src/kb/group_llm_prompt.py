"""
KB 群組 LLM：從模型回覆中解析 JSON（意義／使用方向／威脅對應）。
"""
from __future__ import annotations

import json
import re
from typing import Any

_GROUP_JSON_KEYS = ("meaning", "usage_direction", "threats")


def parse_llm_group_json(text: str) -> dict[str, str]:
    """
    從 LLM 回覆擷取 JSON 物件；支援 ```json ... ``` 包圍。
    回傳鍵至少含 meaning / usage_direction / threats（缺則空字串）。
    """
    out: dict[str, str] = {k: "" for k in _GROUP_JSON_KEYS}
    t = (text or "").strip()
    if not t:
        return out
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t)
    if fence:
        t = fence.group(1).strip()
    start = t.find("{")
    end = t.rfind("}")
    if start < 0 or end <= start:
        return out
    t = t[start : end + 1]
    try:
        raw = json.loads(t)
    except json.JSONDecodeError:
        return out
    if not isinstance(raw, dict):
        return out
    for k in _GROUP_JSON_KEYS:
        v = raw.get(k)
        if isinstance(v, str):
            out[k] = v.strip()
        elif v is not None:
            out[k] = str(v).strip()
    return out


def build_group_llm_prompt(
    *,
    group_key: str,
    group_size: int,
    representative_rel_path: str,
    excerpts: str,
) -> str:
    """繁中指令：要求純 JSON，三鍵與快取 schema 一致。"""
    return (
        "你是資安知識庫編修助理。以下為同一「模板群組」內 KB 文件摘錄。\n"
        f"- group_key: {group_key}\n"
        f"- 檔案數: {group_size}\n"
        f"- 代表路徑: {representative_rel_path}\n\n"
        "請輸出**純 JSON 物件**（不要 markdown 程式碼區塊，不要前後說明文字），鍵名必須完全一致：\n"
        '- "meaning": 此群組知識庫在整體 SOC／編排／回應流程中的**意義**（繁體中文，2-4 句）\n'
        '- "usage_direction": **使用方向**：何時引用、如何與 triage／hunt／回應銜接（繁體中文，2-5 句）\n'
        '- "threats": **威脅對應**：可協助辨識或緩解的威脅類型／情境（繁體中文，2-5 句；若無則寫「無特定威脅對應」）\n\n'
        "摘錄如下：\n---\n"
        f"{excerpts}\n"
        "---\n"
    )
