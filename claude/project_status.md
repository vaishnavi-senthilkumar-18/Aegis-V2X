# Aegis-V2X — Running Project Status Log

This file did not exist in the repository at the time of this entry
(referenced as a "standing open risk" tracker by
`docs/phase2_phase3_reconciliation_2026-09-10.md`, but never actually
present) — recreated here at that same expected path to serve as the
ongoing, dated project-status log going forward. Newest entries first.

---

## 2026-09-19 — Phase 5 checkpoint: TAHS complete, FSDP blocked

**Inspected/recorded by:** Claude Code (Vaishnavi's session)

### Context

Phase 4 (cooperative perception: Data Loader → PointPillars → V2X-ViT,
independent GRU branch, evidence-gated Trust/Criticality estimators) is
closed, validated, committed, and pushed
(`1529d42 feat(phase4): add cooperative perception integration validation`).
This entry records the status of Phase 5 (`ai/twintrust_ap/`,
`digital_twin/`): TAHS and FSDP.

### 1. TAHS — implemented and validated

- `ai/twintrust_ap/tahs.py` now contains a concrete `TAHS(BaseTAHS)` class
  implementing `H_t = H_min + (H_max - H_min) * sigmoid(beta*T_t - gamma*C_t)`,
  with parameters loaded from `configs/model.yaml`'s `tahs:` section
  (`horizon_min=1, horizon_max=10, beta=1.0, gamma=1.0,
  horizon_discretization=[1,2,3,5,8,10]`) and nearest-value snapping onto
  that discrete set. `BaseTAHS` itself is unchanged.
- Public contract used exactly as documented in `docs/interfaces.md`:
  `select_horizon(trust: float, criticality: float) -> int`, both inputs
  in `[0, 1]` — no `DigitalTwinState`, tensors, or Phase 4 outputs
  involved.

### 2. TAHS validation results

- `tests/unit/test_tahs_monotonicity.py` (new): **13/13 passed** —
  covers discrete-set membership, boundary values (0.0/1.0), the
  `docs/interfaces.md` invariant #2 monotonicity property, determinism,
  input-range validation, config-default loading, and hand-verified
  exact outputs.
- Full unit suite (`pytest tests/unit -v`, run in
  `C:\AegisTemp\phase4-validation-venv`): **176 passed, 2 skipped,
  0 failed.** The 2 skips are the pre-existing, intentional
  Mitsuba-unavailable skips (`test_channel_simulator.py`,
  `test_wireless_dataset_generator.py`) — unrelated to Phase 5, left
  untouched.
- Phase 4 regression confirmed passing in the same run (all of
  `test_phase4_pipeline.py`, `test_gru.py`, `test_trust_estimator.py`,
  `test_criticality.py`, `test_v2x_vit.py`, `test_pointpillars.py`).
- No real-data claims made for TAHS — all test inputs are plain
  synthetic floats chosen to exercise the formula/invariants, explicitly
  labeled as such in the test module's docstring.

### 3. FSDP — blocked, not implemented

`ai/twintrust_ap/fsdp.py` remains abstract-only (`BaseFSDP`, `TrustBin`,
`CriticalityBin`, `CommunicationAction` — no concrete subclass).
`ai/twintrust_ap/policy.py` (`BaseTwinTrustAP`, which composes a TAHS +
an FSDP into `JointAdaptiveDecision`) is likewise unimplemented, since it
depends on a working FSDP.

**Exact blocker:**

1. `configs/model.yaml`'s `fsdp.policy_table_path` points to
   `models/fsdp_policy_table.json` — this file **does not exist anywhere
   in the repository** (confirmed by repo-wide search). `models/README.md`
   itself documents it as "Not yet populated (scaffolding only, Phase 1)."
2. The authoritative methodology to generate that table — an "offline
   Pareto optimization pipeline" referenced only as **"Document 2,
   Section 5"** in `ai/twintrust_ap/fsdp.py`'s docstring — is not present
   in the repository in any form. `architecture/offline_pipeline.mmd`
   documents only the pipeline's *stage sequence*
   (`... → TAHS Parameter Learning → Offline Pareto Optimization →
   FSDP Policy Table → Runtime Deployment`), not a runnable procedure.
   `configs/model.yaml`'s `objective_function` gives the optimization
   objective's shape (`J = w1*L + w2*E + w3*O + w4*F - w5*R`) but no cost/
   performance model connecting the 9 `(TrustBin, CriticalityBin)` states
   × 6 actions to those terms — deriving one would require a wireless
   network simulation that doesn't exist in this repo, and real
   CSI/SNR/RSSI/beam-index are `None` in all 5 real scenes, so it cannot
   be derived from real data either.

**Bootstrap table is explicitly excluded:**
`simulation/annotation/decision_labeler.py` contains `BootstrapFSDP`, a
Phase 2 heuristic built to sanity-check dataset trust/criticality labels.
Its own module docstring states it is **"NOT a substitute for the
offline Pareto-optimized policy table that Phase 5 must ultimately
produce"** and **"must not be imported by Phase 5's runtime code as if
it were the final policy."** It has not been used as one here.

No real-data FSDP claims have been made — FSDP has not been executed,
tested, or validated against any data, real or synthetic.

### Required to unblock

Either:
- the authoritative `models/fsdp_policy_table.json` artifact itself
  (produced externally by the intended offline Pareto-optimization
  pipeline), **or**
- the actual methodology document ("Document 2, Section 5" / the Novel
  Algorithm Design spec) plus the cost/performance simulation data it
  depends on, so the table can be legitimately generated inside this
  repository.

No repository-internal workaround exists — this was confirmed by a
dedicated read-only audit (repo-wide search for `fsdp_policy_table.json`,
`Pareto`, `Document 2`/`Section 5`, and all `architecture/*.mmd` /
`configs/model.yaml` cross-references) before this entry was written.
