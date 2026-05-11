"""
TensorStress — Computational Stress Tensor Analysis & Physics-Constrained ML Toolkit.

Domain-neutral framework for:
- Moment tensor mathematics (source mechanics)
- Coulomb Failure Stress computation
- Green's function stress superposition
- Weighted ensemble regression with physical law enforcement
- Comprehensive ML model auditing
"""

from tensorstress.moment_tensor import (mt_from_sdr, receiver_geometry, basis_mt_decomposition)
from tensorstress.cfs import (compute_cfs, analytical_stress_field)
from tensorstress.weighted_ensemble import WeightedEnsemble
from tensorstress.audit import ModelAuditor

__version__ = "1.0.0"
