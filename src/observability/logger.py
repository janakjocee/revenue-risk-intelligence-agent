from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def log_agent_run(record: dict[str, Any], log_path: str | Path = "data/processed/agent_runs.jsonl") -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = {"run_id": record.get("run_id") or str(uuid.uuid4()), "timestamp": datetime.now(timezone.utc).isoformat(), **record}
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(enriched) + "\n")


def read_agent_logs(log_path: str | Path = "data/processed/agent_runs.jsonl") -> list[dict[str, Any]]:
    path = Path(log_path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_feedback(
    record: dict[str, Any],
    feedback_path: str | Path = "data/processed/feedback.jsonl",
) -> dict[str, Any]:
    path = Path(feedback_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(enriched) + "\n")
    return enriched


def read_feedback(feedback_path: str | Path = "data/processed/feedback.jsonl") -> list[dict[str, Any]]:
    path = Path(feedback_path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def feedback_summary(feedback_path: str | Path = "data/processed/feedback.jsonl") -> dict[str, Any]:
    rows = read_feedback(feedback_path)
    if not rows:
        return {"feedback_count": 0, "average_rating": None, "positive_share": None}
    ratings = [float(row["rating"]) for row in rows if row.get("rating") is not None]
    positive = [rating for rating in ratings if rating >= 4]
    return {
        "feedback_count": len(rows),
        "average_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
        "positive_share": round(len(positive) / len(ratings), 3) if ratings else None,
    }
