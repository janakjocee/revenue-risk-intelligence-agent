from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def log_agent_run(record: dict[str, Any], log_path: str | Path = "data/processed/agent_runs.jsonl") -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(enriched) + "\n")


def read_agent_logs(log_path: str | Path = "data/processed/agent_runs.jsonl") -> list[dict[str, Any]]:
    path = Path(log_path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

