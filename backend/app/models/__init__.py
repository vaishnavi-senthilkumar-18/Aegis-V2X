"""ORM model registry.

Importing this package registers every model class on `Base.metadata`,
which Alembic's `env.py` relies on for autogenerate and which
`Base.metadata.create_all()` relies on in tests.
"""

from app.models.criticality import CriticalityRecord
from app.models.decision import Decision
from app.models.experiment import Experiment
from app.models.frame import Frame
from app.models.scene import Scene, Vehicle
from app.models.trust import TrustRecord

__all__ = [
    "Scene",
    "Vehicle",
    "Frame",
    "TrustRecord",
    "CriticalityRecord",
    "Decision",
    "Experiment",
]
