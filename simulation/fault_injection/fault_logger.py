
"""JSON Lines logging for Aegis-V2X fault-injection events."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class FaultLogger:
    """Append structured fault events to a JSON Lines file."""

    def __init__(self, log_path: str | Path = "logs/fault_injection.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        *,
        component: str,
        fault: str,
        parameters: dict[str, Any],
        outcome: dict[str, Any],
        scenario_id: str = "unspecified",
    ) -> dict[str, Any]:
        """Write one fault event and return the recorded event."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scenario_id": scenario_id,
            "component": component,
            "fault": fault,
            "parameters": parameters,
            "outcome": outcome,
        }

        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(event, sort_keys=True) + "\n")

        return event
