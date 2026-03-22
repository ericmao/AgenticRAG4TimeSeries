"""
單檔 KB LLM：從模型回覆解析 JSON（summary 一句／數句繁中概要）。
"""
from __future__ import annotations

import json
import re


def parse_file_summary_json(text: str) -> str:
    """從 LLM 回覆擷取 {"summary": "..."}；支援 ```json ... ```。"""
    t = (text or "").strip()
    if not t:
        return ""
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t)
    if fence:
        t = fence.group(1).strip()
    start = t.find("{")
    end = t.rfind("}")
    if start < 0 or end <= start:
        return ""
    t = t[start : end + 1]
    try:
        raw = json.loads(t)
    except json.JSONDecodeError:
        return ""
    if not isinstance(raw, dict):
        return ""
    v = raw.get("summary")
    if isinstance(v, str):
        return v.strip()
    if v is not None:
        return str(v).strip()
    return ""


def build_file_llm_prompt(*, rel_path: str, excerpt: str) -> str:
    return (
        "你是資安／SOC 知識庫助理。以下為**單一** KB 檔摘錄。\n"
        f"- 相對路徑: {rel_path}\n\n"
        "請用繁體中文寫 **1～3 句概要**，說明此文件用途、適用情境與重點（勿逐字抄標題）。\n"
        "輸出**純 JSON 物件**（不要 markdown 程式碼區塊，不要前後說明），鍵名必須為：\n"
        '- "summary": 字串\n\n'
        "摘錄如下：\n---\n"
        f"{excerpt}\n"
        "---\n"
    )
