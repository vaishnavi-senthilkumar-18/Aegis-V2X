"""
Scenario catalog: tracks every fault-injected scenario run that has been generated, so
Phase 7 evaluation can look up "give me every High-severity packet_loss run" reproducibly.

Stored as a flat JSON file for now (simple, human-readable, diff-friendly in git). If this
needs to scale to thousands of entries, migrate to Logapriya's PostgreSQL backend via the
`experiments` table instead of inventing a second storage system.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .schemas import ScenarioFaultPlan


@dataclass
class CatalogEntry:
    """One row in the scenario catalog: a fault plan plus its execution outcome."""

    entry_id: str
    scenario_id: str
    seed: int
    fault_plan: dict  # ScenarioFaultPlan.model_dump()
    num_frames_in: int
    num_frames_out: int
    num_frames_dropped: int
    notes: str = ""


class ScenarioCatalog:
    """Append-only JSON-backed catalog of fault-injection runs."""

    def __init__(self, catalog_path: str | Path):
        self.catalog_path = Path(catalog_path)
        if not self.catalog_path.exists():
            self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_entries([])

    def _read_entries(self) -> list[dict]:
        with open(self.catalog_path, "r") as f:
            return json.load(f)

    def _write_entries(self, entries: list[dict]) -> None:
        with open(self.catalog_path, "w") as f:
            json.dump(entries, f, indent=2)

    def add_entry(
        self,
        plan: ScenarioFaultPlan,
        num_frames_in: int,
        num_frames_out: int,
        num_frames_dropped: int,
        notes: str = "",
    ) -> CatalogEntry:
        """Record one fault-injection run in the catalog."""
        entries = self._read_entries()
        entry_id = f"{plan.scenario_id}__seed{plan.seed}__{len(entries):04d}"
        entry = CatalogEntry(
            entry_id=entry_id,
            scenario_id=plan.scenario_id,
            seed=plan.seed,
            fault_plan=plan.model_dump(),
            num_frames_in=num_frames_in,
            num_frames_out=num_frames_out,
            num_frames_dropped=num_frames_dropped,
            notes=notes,
        )
        entries.append(asdict(entry))
        self._write_entries(entries)
        return entry

    def all_entries(self) -> list[dict]:
        return self._read_entries()

    def find_by_scenario(self, scenario_id: str) -> list[dict]:
        return [e for e in self._read_entries() if e["scenario_id"] == scenario_id]

    def find_by_fault_type(self, fault_type: str) -> list[dict]:
        matches = []
        for entry in self._read_entries():
            fault_types_in_entry = {f["type"] for f in entry["fault_plan"]["faults"]}
            if fault_type in fault_types_in_entry:
                matches.append(entry)
        return matches
