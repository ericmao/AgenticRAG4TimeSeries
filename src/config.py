"""
Configuration loaded from environment. Use env.example as template; never commit .env.
"""
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

# Load .env from repo root (parent of src/)
_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")


class Config(BaseModel):
    """App config from env with sensible defaults. Fail fast when RUN_MODE != dry_run and required vars missing."""

    OPENAI_API_KEY: str = ""
    OPENCTI_URL: str = ""
    OPENCTI_TOKEN: str = ""
    KB_PATH: str = "kb/"
    # off | merge | db_only — 見 src/kb/loader.py
    KB_DB_MODE: str = "off"
    RUN_MODE: str = "dry_run"
    PROMPT_VERSION: str = "v0.1"

    # LLM: default local Ollama
    LLM_BACKEND: str = "ollama"
    LLM_OLLAMA_PRIMARY: str = "http://127.0.0.1:11434"
    LLM_MODEL: str = "llama3:latest"
    LLM_TIMEOUT: int = 120
    LLM_TEMPERATURE: float = 0.1

    # Layer C optional: time-series signals and LangChain analysis
    USE_TIME_SERIES_SIGNALS: bool = False
    USE_LANGCHAIN_FOR_ANALYSIS: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        import os
        def _bool(key: str, default: bool) -> bool:
            v = os.environ.get(key, "").strip().lower()
            if v in ("1", "true", "yes"):
                return True
            if v in ("0", "false", "no"):
                return False
            return default
        return cls(
            OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY", "").strip(),
            OPENCTI_URL=os.environ.get("OPENCTI_URL", "").strip(),
            OPENCTI_TOKEN=os.environ.get("OPENCTI_TOKEN", "").strip(),
            KB_PATH=os.environ.get("KB_PATH", "kb/").strip() or "kb/",
            KB_DB_MODE=(
                os.environ.get("KB_DB_MODE", "off").strip().lower() or "off"
            ),
            RUN_MODE=os.environ.get("RUN_MODE", "dry_run").strip() or "dry_run",
            PROMPT_VERSION=os.environ.get("PROMPT_VERSION", "v0.1").strip() or "v0.1",
            LLM_BACKEND=os.environ.get("LLM_BACKEND", "ollama").strip() or "ollama",
            LLM_OLLAMA_PRIMARY=os.environ.get("LLM_OLLAMA_PRIMARY", "http://127.0.0.1:11434").strip() or "http://127.0.0.1:11434",
            LLM_MODEL=os.environ.get("LLM_MODEL", "llama3:latest").strip() or "llama3:latest",
            LLM_TIMEOUT=int(os.environ.get("LLM_TIMEOUT", "120").strip() or "120"),
            LLM_TEMPERATURE=float(os.environ.get("LLM_TEMPERATURE", "0.1").strip() or "0.1"),
            USE_TIME_SERIES_SIGNALS=_bool("USE_TIME_SERIES_SIGNALS", False),
            USE_LANGCHAIN_FOR_ANALYSIS=_bool("USE_LANGCHAIN_FOR_ANALYSIS", False),
        )

    def validate_required_for_run(self) -> None:
        """Raise clear error if required env vars are missing when RUN_MODE != dry_run."""
        if self.RUN_MODE == "dry_run":
            return
        missing: list[str] = []
        if not self.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if not self.OPENCTI_URL:
            missing.append("OPENCTI_URL")
        if not self.OPENCTI_TOKEN:
            missing.append("OPENCTI_TOKEN")
        if missing:
            raise SystemExit(
                f"RUN_MODE={self.RUN_MODE} requires these env vars to be set (copy env.example to .env): {', '.join(missing)}"
            )


def get_config() -> Config:
    """Load and return config. Call validate_required_for_run() when starting a non–dry_run."""
    return Config.from_env()
