"""
Export Layer C Pydantic models to JSON Schema under outputs/schemas/*.json.
Validates tests/samples/episode.json against Episode when run as __main__.
"""
import json
import sys
from pathlib import Path

# Repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMAS_DIR = _REPO_ROOT / "outputs" / "schemas"
_SAMPLES_DIR = _REPO_ROOT / "tests" / "samples"


def _export_schemas() -> None:
    """Write JSON schemas for all contract models to outputs/schemas/*.json."""
    from src.contracts import (
        AgentOutput,
        Episode,
        EvidenceItem,
        EvidenceSet,
        HuntPlan,
        Hypothesis,
        PolicyAction,
        PolicyRule,
        ResponsePlan,
    )

    _SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)
    models = [
        ("Episode", Episode),
        ("Hypothesis", Hypothesis),
        ("EvidenceItem", EvidenceItem),
        ("EvidenceSet", EvidenceSet),
        ("AgentOutput", AgentOutput),
        ("HuntPlan", HuntPlan),
        ("ResponsePlan", ResponsePlan),
        ("PolicyAction", PolicyAction),
        ("PolicyRule", PolicyRule),
    ]
    for name, model in models:
        path = _SCHEMAS_DIR / f"{name}.json"
        schema = model.model_json_schema()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)
        print(f"  wrote {path.relative_to(_REPO_ROOT)}")


def validate_episode_sample() -> None:
    """Load tests/samples/episode.json and validate against Episode."""
    from src.contracts import Episode

    path = _SAMPLES_DIR / "episode.json"
    if not path.exists():
        print(f"  skip: {path.relative_to(_REPO_ROOT)} not found")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    Episode.model_validate(data)
    print(f"  validated {path.relative_to(_REPO_ROOT)} against Episode")


def validate_hypothesis_sample() -> None:
    """Optional: validate tests/samples/hypothesis.json against Hypothesis."""
    from src.contracts import Hypothesis

    path = _SAMPLES_DIR / "hypothesis.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    Hypothesis.model_validate(data)
    print(f"  validated {path.relative_to(_REPO_ROOT)} against Hypothesis")


def run() -> None:
    """Export schemas and validate sample episode (and optional hypothesis)."""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    print("Exporting JSON schemas...")
    _export_schemas()
    print("Validating sample episode...")
    validate_episode_sample()
    validate_hypothesis_sample()
    print("Done.")


if __name__ == "__main__":
    run()
