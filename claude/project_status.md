# Aegis-V2X — Running Project Status Log

This file did not exist in the repository at the time of this entry
(referenced as a "standing open risk" tracker by
`docs/phase2_phase3_reconciliation_2026-09-10.md`, but never actually
present) — recreated here at that same expected path to serve as the
ongoing, dated project-status log going forward. Newest entries first.

---

## 2026-09-19 (latest) — GDCA formulation corrected (Case Y / Case G fixes)

**Recorded by:** Claude Code (Vaishnavi's session)

### Defect (from the prior design-review checkpoint)

The original GDCA formula (`z_t = residual_norm / (rolling_std + eps)`)
had two confirmed defects: (1) **Case Y** — plain rolling standard
deviation has a 0% breakdown point, so a single unrelated large
residual anywhere in the recent window could inflate `sigma_t` enough
to score a separately bad *current* residual with artificially high
authority (numerically confirmed: `0.000000 -> 0.819871` for the exact
same current residual, purely from one unrelated past outlier); (2)
**Case G** — pooling all state dimensions into one Euclidean norm before
normalization let whichever dimension had the largest raw numeric units
dominate, regardless of physical meaning.

### Fix 1 — bias/spread separation via MAD (Case Y)

`ai/twintrust_ap/gdca.py` gained `residual_level_and_spread()` /
`robust_spread()`: Median Absolute Deviation (scaled ×1.4826, the
standard Gaussian-consistent conversion), replacing rolling standard
deviation as GDCA's recommended uncertainty source. MAD's 50% breakdown
point means one outlier cannot manufacture an inflated spread. Verified:
same Case Y scenario now gives `0.000000` authority for the contaminated
window, matching a clean window with the identical current residual
(previously `0.819871` vs `0.000000` — a stark mismatch; now both
`≈0.000000`). Critically, **the raw current residual is still normalized
against ZERO, not against the window's own median** — a perfectly
consistent-but-wrong predictor (residual always exactly 3.0) still
correctly collapses to `authority≈0` (MAD of a constant window is 0),
preserving the invariant that "consistent" must never be read as
"accurate" (Case C, unchanged and re-verified).

### Fix 2 — per-dimension normalization (Case G)

New `standardize_residual_dimensions()` (per-dimension `z_i =
residual_i / (spread_i + eps)`, each dimension normalized by its OWN
independently-estimated spread) + `combine_standardized_residuals()`
(simple, weight-free RMS combination — explicitly no learned weights, no
covariance matrix, no Mahalanobis distance, per the design review's
Part 3 finding that no repository evidence justifies that machinery) +
`compute_authority_from_dimensional_evidence()` (wires both together,
then reuses the *same* `exp(-0.5 z²)` mapping — only `z`'s construction
changed, not the authority mapping the design review found sound).
Verified: a small-raw-unit dimension that is anomalous *relative to its
own spread* now correctly pulls authority down even when a
large-raw-unit dimension would have dominated the old pooled norm, and
vice versa (dominance now tracks standardized anomaly, not raw units).

### Backward compatibility

`compute_residual`, `residual_norm`, `rolling_uncertainty`, and
`compute_authority_from_evidence` are **unchanged** — all 30 pre-existing
tests pass without modification. They remain valid for their original,
narrower case: dimensions that already share real, comparable units
(e.g. the position-only x/y/z residual the real-data demo uses).
`rolling_uncertainty` (std-based) is kept as a general-purpose utility;
it is simply no longer GDCA's recommended uncertainty source.

### Tests

`tests/unit/test_gdca.py` grew from 30 to **53 tests** — added:
`robust_spread`/`residual_level_and_spread` correctness (5),
`standardize_residual_dimensions`/`combine_standardized_residuals`/
`compute_authority_from_dimensional_evidence` correctness (7), and the
mandatory adversarial regressions: Case Y (3 tests — old-formulation
defect pinned, corrected formulation verified, direct old-vs-corrected
comparison), Case G (3 tests — pooled-norm defect pinned, corrected
non-domination verified, converse small-raw/large-standardized case
verified), Case C (2 tests — consistent-but-wrong stays ~0, consistent-
and-accurate stays 1), Case E (1 test — tiny genuine residual doesn't
collapse to the ~1e-9 floor), Case F (2 tests — no NaN/Inf/negative
under a regime change, and the documented rolling-window lag limitation
pinned explicitly rather than silently hidden).

**Result: 53/53 passed.**

### Real-data mechanism check (corrected formulation)

Re-ran the same mechanism check as the prior checkpoint — real 92-frame
position sequence, `Vehicle147`, `straight_road_dense_clear_day_Scene00`
(`scene_id=6b08be32-e041-4ab4-861c-99090df77ffb`), same explicitly-labeled
naive persistence baseline (no trained predictor exists in this repo —
unchanged limitation). 91 residual comparisons, rolling window = 5.

- **Authority range:** `[0.00000000, 0.99931920]`
- **Stationary behavior:** once the vehicle settles (~frame 121537
  onward), authority is correctly near-maximum (`0.999319` at frame
  121537, `0.994814` at 121547).
- **Acceleration behavior:** once the vehicle accelerates (frame ≳121600
  onward), the naive baseline's constant-position assumption breaks down
  and authority correctly collapses to `0.000000` and stays there through
  frame 122407 — consistent with Case C's "consistent-but-wrong" finding:
  the residual becomes large AND steady (near-constant velocity motion),
  so MAD-spread shrinks toward the numerical floor and the persistently
  bad baseline is correctly never rewarded for its consistency.
- **NaN/Inf occurrences:** 0.

This is a mechanism check only (see the design review's Part 8) — not
predictor validation, not calibration, not a claim about Digital-Twin
prediction accuracy.

### Regression

- `tests/unit/test_phase4_pipeline.py`: **8/8 passed** (unchanged).
- `tests/unit/test_tahs_monotonicity.py`: **13/13 passed** (unchanged).
- Full suite (`pytest tests/unit -v`): **229 passed, 2 skipped
  (pre-existing Mitsuba skips), 0 failed** (206 prior + 53 GDCA − 30
  superseded count = 229; i.e. +23 net new GDCA tests this round).
- `git diff --check`: clean.

### Still true, unchanged

- No trained Digital-Twin predictor exists in this repository.
- No GDCA calibration labels exist (Part 8 of the design review).
- GDCA is **not** wired into TAHS or any runtime loop.
- FSDP remains blocked (missing `models/fsdp_policy_table.json` and its
  offline Pareto-optimization methodology — see the 2026-09-19 entry
  above).
- Trust and GDCA remain separate, evidence-disjoint signals (Part 5 of
  the design review) — this checkpoint did not touch that boundary.

### Remaining limitations

- No principled role for `sync_age_seconds` yet (unchanged).
- Temporal/multi-step authority tracking still not implemented — the
  rolling-window lag after a regime change (Case F) is now explicitly
  documented and regression-tested, not silently accepted as solved.
- No calibration method exists for mapping `A_t` to any operational
  outcome.
- Real-data check remains mechanism-only (Part 8, level 1) — predictor
  validation, calibration, and end-to-end validation all remain future
  work requiring evidence this repository does not yet have.

## 2026-09-19 (later) — GDCA (Graded Digital-Twin Consistency Authority) added

**Recorded by:** Claude Code (Vaishnavi's session)

### Purpose

GDCA is a new, third Phase 5 mechanism converting Digital-Twin
prediction/observation consistency into a continuous, bounded
**authority score** A_t ∈ [0,1] — distinct from both Trust (which
estimates *communication-reliability* confidence from wireless/sync
evidence) and Criticality (situational urgency). Per the intended
architecture, GDCA sits upstream of TAHS/FSDP as a sibling of
Criticality, not a child of Trust:

```
Digital Twin -> prediction/observation consistency -> GDCA -> Authority A_t
                                                                    +
                                              (separately) Criticality C_t
                                                                    ↓
                                                          TAHS / FSDP
```

### Mathematical definition

```
r_t   = observed_state - predicted_state                 (innovation/residual)
z_t   = ||r_t|| / (sigma_t + eps)                          (uncertainty-normalized)
A_t   = exp(-0.5 * z_t^2)                                   in (0, 1]
```

`sigma_t` (the uncertainty scale) is a windowed standard deviation of
recent residual norms — the same technique
`simulation/annotation/future_channel_labeler.py`'s `_rolling_uncertainty`
already uses for CSI magnitude, generalized here to any residual-norm
sequence. This form is the same idea as Normalized Innovation Squared
(NIS) consistency checks in Kalman filtering — not an invented concept.

**Deliberately excluded:** `sync_age_seconds` (its correct sign/role is
unspecified anywhere in the repo) and multi-step temporal tracking (an
EWMA of A_t across a sequence) — both are documented as future,
evidence-backed extensions, not silently added or silently omitted.
Criticality is never an input to GDCA.

### Files changed

- `ai/twintrust_ap/gdca.py` (new) — `BaseGDCAEstimator` (2-argument
  `estimate(observed_state, predicted_state)`, deliberately different
  from Trust/Criticality's 1-argument `estimate(state)` since GDCA
  compares two states), `GDCAEstimator` (concrete; `estimate()` always
  raises `MissingGDCAEvidenceError` — a single state pair cannot supply
  the required uncertainty *window*, mirroring
  `TwinTrustEstimator.estimate`'s own structural gap), and the testable
  math: `compute_residual`, `residual_norm`, `rolling_uncertainty`,
  `compute_authority_from_evidence`.
- `tests/unit/test_gdca.py` (new) — 30 tests.

TAHS (`ai/twintrust_ap/tahs.py`) was NOT modified. FSDP
(`ai/twintrust_ap/fsdp.py`, `ai/twintrust_ap/policy.py`) was NOT
modified and remains unimplemented (see the 2026-09-19 entry above —
still blocked on the missing `models/fsdp_policy_table.json` and its
offline Pareto-optimization methodology). Phase 4 and the dashboard were
not touched.

### Validation

- `tests/unit/test_gdca.py`: **30/30 passed**.
- Full suite (`pytest tests/unit -v`,
  `C:\AegisTemp\phase4-validation-venv`): **206 passed, 2 skipped
  (pre-existing Mitsuba skips), 0 failed** — includes Phase 4 (8/8) and
  TAHS (13/13) regression, both still passing unchanged.
- `git diff --check`: clean.

### Real-data demonstration (honest, not fabricated)

Used `ai/perception/data_loader.Phase4DataLoader` against the live
backend to pull the real, full position sequence (92 frames) for
`Vehicle147` in `straight_road_dense_clear_day_Scene00`
(`scene_id=6b08be32-e041-4ab4-861c-99090df77ffb`). Since no trained/real
Digital-Twin position predictor exists anywhere in this repository, the
"predicted state" used is an explicitly-labeled **naive persistence
(zero-order-hold) baseline** — `predicted_state(t) = observed_state(t-1)`
— the same kind of honestly-labeled placeholder baseline
`future_channel_labeler.py` already uses for CSI. This is NOT presented
as a trained Digital-Twin prediction.

Representative real results (real `position_x, position_y, position_z`
throughout, rolling window = 5):

| Frame | Observed (x, y, z) | Naive-baseline predicted (x, y, z) | residual_norm | uncertainty | Authority |
|---|---|---|---|---|---|
| 121537 | (4.310, -160.060, 0.014) | (4.310, -160.060, 0.015) | 0.0017 | 0.0522 | 0.9995 |
| 121577 | (4.310, -160.200, 0.031) | (4.310, -160.095, 0.030) | 0.1052 | 0.0389 | 0.0259 |
| 122407 | (23.346, -295.331, 0.034) | (22.184, -293.188, 0.034) | 2.4381 | 0.0001 | ~0.0000 |

Honest finding: while the vehicle is nearly stationary (~frame 121537),
the naive baseline tracks well and GDCA correctly reports near-maximum
authority. Once the vehicle accelerates (frame ≳121600 onward), the
zero-order-hold assumption breaks down and authority correctly collapses
toward 0 — GDCA is behaving exactly as a consistency-authority mechanism
should: it does not report high authority for a baseline predictor that
is, in fact, wrong. This demonstrates the mechanism operating correctly
on real, non-fabricated position data; it is NOT a claim that the
Digital Twin accurately predicts vehicle motion — no such predictor
exists yet.

### Synthetic validation status

All 30 `test_gdca.py` tests use plain synthetic floats/tuples to
exercise the formula's boundedness, determinism, monotonicity, and
input validation — explicitly labeled as such in the test module's
docstring, per the same convention `test_tahs_monotonicity.py` and
`test_gru.py` already use.

### TAHS compatibility

Not integrated into TAHS this round. `TAHS.select_horizon(trust:
float, criticality: float)` takes a named `trust` argument; GDCA
produces a differently-scoped `authority` score. Structurally, a float
in `[0,1]` COULD be passed positionally where `trust` is expected
without a code change, but doing so would silently redefine what TAHS's
`trust` parameter means (semantically becoming
"Trust-or-Authority-whichever-was-passed") without any interface change
recording that. No such call was made. This is flagged as an open
design question, not resolved.

### FSDP status

Unchanged: still abstract-only, still blocked on the missing
`models/fsdp_policy_table.json` and its offline Pareto-optimization
methodology (see entry above). GDCA was not connected to FSDP in any
way — this is documented only as a future architectural proposal
(Authority + Criticality → FSDP), never executed.

### Limitations

- No principled role for `sync_age_seconds` yet.
- No multi-step/temporal authority tracking yet (single-step only).
- Real-data demonstration uses a naive persistence baseline, not a
  trained predictor (none exists in this repository).
- Not yet wired into TAHS or any runtime execution loop.

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
