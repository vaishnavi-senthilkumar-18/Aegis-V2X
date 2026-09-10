"""Custom Prometheus domain metrics for Aegis-V2X.

Standard HTTP metrics (request count, latency, in-progress requests) are
provided automatically by `prometheus-fastapi-instrumentator` in
`app.main`. The five metrics below capture research-domain signals that a
generic HTTP instrumentator cannot know about, and are recorded from
wherever the underlying event actually happens (API handlers *and* the
synthetic data generator) so dashboards stay accurate regardless of
ingestion path — see `docs/backend_api_documentation.md` §6 for the bug
this fixes (metrics gap when synthetic data bypassed the API layer).
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

trust_score_histogram = Histogram(
    "aegis_trust_score",
    "Distribution of calibrated Digital Twin trust probabilities T_t.",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

criticality_score_histogram = Histogram(
    "aegis_criticality_score",
    "Distribution of scene criticality scores C_t.",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

prediction_horizon_gauge = Gauge(
    "aegis_prediction_horizon_latest",
    "Most recently selected prediction horizon H_t (frames).",
)

fsdp_decisions_counter = Counter(
    "aegis_fsdp_decisions_total",
    "Count of FSDP actions taken, labeled by action.",
    labelnames=("action",),
)

ingested_frames_counter = Counter(
    "aegis_ingested_frames_total",
    "Count of frames ingested, labeled by source (real vs synthetic).",
    labelnames=("source",),
)
