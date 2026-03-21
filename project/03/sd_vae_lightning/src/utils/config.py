import json
from pathlib import Path
from typing import Any, Dict

from pytorch_lightning.utilities.rank_zero import rank_zero_only


def load_json_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    return cfg


@rank_zero_only
def save_json_config(config: Dict[str, Any], save_path: str) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
