# Aegis-V2X — Interface Reference (Phase 1)

All interfaces below are Python `abc.ABC` classes with `@abstractmethod`
signatures only — no logic is implemented in Phase 1, per the Master
Project Instructions ("do not implement AI models yet").

Every method is documented with the `Purpose / Parameters / Returns /
Example` docstring convention mandated in
`Title07_Project_Structure_and_Software_Engineering_Guide.docx`, Section 6.

## Data Contracts

- `digital_twin.state.DigitalTwinState` — frozen dataclass, DT_t
- `digital_twin.state.ChannelState`, `MobilityState`, `EnvironmentalContext` — sub-components of DT_t
- `ai.twintrust_ap.fsdp.TrustBin`, `CriticalityBin`, `CommunicationAction` — enums
- `ai.twintrust_ap.policy.JointAdaptiveDecision` — frozen dataclass, TwinTrust-AP output

## Behavioral Contracts

- `digital_twin.interfaces.BaseDigitalTwinManager`
  - `get_current_state() -> DigitalTwinState`
  - `synchronize(ground_truth_state) -> DigitalTwinState`
  - `advance(dt_seconds) -> DigitalTwinState`
- `ai.trust_estimator.base.BaseTrustEstimator`
  - `estimate(state) -> float`
  - `calibration_error(predictions, outcomes) -> float`
- `ai.criticality.base.BaseCriticalityEstimator`
  - `estimate(state) -> float`
- `ai.twintrust_ap.tahs.BaseTAHS`
  - `select_horizon(trust, criticality, authority=0.0) -> int`
- `ai.twintrust_ap.fsdp.BaseFSDP`
  - `discretize(trust, criticality) -> (TrustBin, CriticalityBin)`
  - `lookup_action(trust_bin, criticality_bin) -> CommunicationAction`
  - `decide(trust, criticality) -> CommunicationAction` (composed, concrete)
- `ai.twintrust_ap.policy.BaseTwinTrustAP`
  - `decide(trust, criticality) -> JointAdaptiveDecision`
- `ai.pointpillars.base.BasePointPillars`
  - `extract_features(lidar_point_cloud) -> Any`
- `ai.v2x_vit.base.BaseV2XViT`
  - `fuse(per_vehicle_features) -> Any`
- `ai.gru.base.BaseGRUPredictor`
  - `predict(historical_sequence, horizon) -> Any`

## Invariants Implementers Must Preserve

1. **Boundedness** — trust and criticality outputs must lie in `[0, 1]`.
2. **Monotonicity** (TAHS) — extended in Phase 5 to all three inputs
   (`authority` is GDCA's Authority score, `ai/twintrust_ap/gdca.py`;
   defaults to 0.0, the additive identity, so two-input callers are
   unaffected):
   - for fixed criticality and authority, `T1 > T2 => H1 >= H2`
   - for fixed trust and authority, `C1 > C2 => H1 <= H2`
   - for fixed trust and criticality, `A1 > A2 => H1 >= H2`
3. **Determinism** — FSDP must return the same action for the same
   `(trust_bin, criticality_bin)` pair.

These map directly to `03_Mathematical_Formulation.docx`, Section 11, and
should each have a corresponding property-based unit test written in
Phase 4/5 (`tests/unit/test_trust_estimator.py`,
`tests/unit/test_tahs_monotonicity.py`, `tests/unit/test_fsdp_determinism.py`).
