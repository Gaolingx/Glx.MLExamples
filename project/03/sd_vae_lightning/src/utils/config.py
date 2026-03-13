import json
from pathlib import Path
from typing import Any, Dict


def load_json_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    return cfg
