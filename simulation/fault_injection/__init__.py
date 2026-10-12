"""
Aegis-V2X Phase 6 (Track A) — Scenario & Fault Injection module.

Public API:
    from simulation.fault_injection import FaultInjector, ScenarioFaultPlan, FaultConfig
    from simulation.fault_injection import ScenarioCatalog

Owner: Haridharani
"""

from .injector import FaultInjector, InjectionResult
from .scenario_catalog import ScenarioCatalog
from .schemas import FaultCategory, FaultConfig, FaultType, ScenarioFaultPlan
from .severity import SeverityBand, severity_band, validate_severity

__all__ = [
    "FaultInjector",
    "InjectionResult",
    "ScenarioCatalog",
    "FaultCategory",
    "FaultConfig",
    "FaultType",
    "ScenarioFaultPlan",
    "SeverityBand",
    "severity_band",
    "validate_severity",
]
