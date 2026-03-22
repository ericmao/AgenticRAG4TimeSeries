"""從 repo 根目錄 .env 強制套用 WAZUH_*（非空時），避免 Docker 空字串阻擋 load_dotenv。"""
from __future__ import annotations

import os
from pathlib import Path


def apply_wazuh_env_from_dotenv(repo_root: Path) -> None:
    p = repo_root / ".env"
    if not p.is_file():
        return
    try:
        from dotenv import dotenv_values

        vals = dotenv_values(p)
    except ImportError:
        return
    for key in ("WAZUH_INDEXER_URL", "WAZUH_INDEXER_USERNAME", "WAZUH_INDEXER_PASSWORD", "WAZUH_VERIFY_SSL"):
        v = vals.get(key)
        if v is not None and str(v).strip() != "":
            os.environ[key] = str(v).strip()
