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
    RUN_MODE: str = "dry_run"
    PROMPT_VERSION: str = "v0.1"

    @classmethod
    def from_env(cls) -> "Config":
        import os
        return cls(
            OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY", "").strip(),
            OPENCTI_URL=os.environ.get("OPENCTI_URL", "").strip(),
            OPENCTI_TOKEN=os.environ.get("OPENCTI_TOKEN", "").strip(),
            KB_PATH=os.environ.get("KB_PATH", "kb/").strip() or "kb/",
            RUN_MODE=os.environ.get("RUN_MODE", "dry_run").strip() or "dry_run",
            PROMPT_VERSION=os.environ.get("PROMPT_VERSION", "v0.1").strip() or "v0.1",
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
