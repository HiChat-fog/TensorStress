"""
Coulomb Failure Stress (CFS) Computation.

Two methods:
1. Analytical stress field from point-source (Okada-type scaling)
   Fast approximation using seismic source parameters.
2. Green's function superposition for arbitrary receiver mechanisms.
   Linear superposition of pre-computed basis stress fields.

CFS = shear_stress + friction_coefficient * normal_stress
"""

import numpy as np
from tensorstress.moment_tensor import (
    mt_from_sdr, receiver_geometry, cfs_from_stress_tensor
)


def analytical_stress_field(magnitude, depth_km, strike, dip, rake,
                            friction=0.4, shear_modulus=3.0e10,
                            stress_drop=3.0e6):
    """Analytical stress field from a point-source in elastic half-space.

    Uses scaling relations:
    - Seismic moment: M0 = 10^(1.5*M + 9.1) [Nm]
    - Source radius: r = (7*M0 / 16*stress_drop)^(1/3)
    - Stress at distance R: sigma ~ shear_modulus * slip * area / R^3

    Also computes CFS on 5 canonical receiver types (thrust, normal,
    strike-slip, oblique) and returns the maximum.

    Returns: dict of CFS values (in MPa) for each receiver type + max
    """
    M0 = 10 ** (1.5 * magnitude + 9.1)
    r = (7.0 * M0 / (16.0 * stress_drop)) ** (1.0 / 3.0)
    area = np.pi * r * r
    slip = M0 / (shear_modulus * area)

    # Observation distance: at least 2 source radii or depth
    obs_dist = max(2.0 * r, depth_km * 1000.0)
    R = np.sqrt(obs_dist ** 2 + (depth_km * 1000.0) ** 2)
    if R < 1.0:
        R = 1.0

    stress_scale = shear_modulus * slip * area / (R ** 3)
    mt_norm = mt_from_sdr(strike, dip, rake)
    sigma = stress_scale * mt_norm

    # Evaluate CFS on canonical receiver mechanisms
    receiver_types = {
        'thrust':   [(0, 30, 90), (90, 30, 90), (180, 30, 90), (270, 30, 90)],
        'normal':   [(0, 60, -90), (90, 60, -90), (180, 60, -90), (270, 60, -90)],
        'strike':   [(0, 90, 0), (45, 90, 0), (90, 90, 0), (135, 90, 0)],
        'oblique1': [(30, 45, 45), (120, 45, 45), (210, 45, 45), (300, 45, 45)],
        'oblique2': [(60, 45, -45), (150, 45, -45), (240, 45, -45), (330, 45, -45)],
    }

    results = {}
    for ftype, candidates in receiver_types.items():
        best = -1e30
        for sr, dr, rr in candidates:
            n, sv = receiver_geometry(sr, dr, rr)
            tau = n @ sigma @ sv
            sigma_n = n @ sigma @ n
            cfs_val = tau + friction * sigma_n
            if cfs_val > best:
                best = cfs_val
        results[f'cfs_{ftype}'] = float(best / 1e6)

    results['cfs_max'] = max(results.values())
    return results


def compute_cfs(stress_6d, receiver_strike, receiver_dip, receiver_rake,
                mu_friction=0.4):
    """Compute CFS from a 6-component stress tensor (Voigt notation).

    stress_6d: (Sxx, Syy, Szz, Sxy, Syz, Szx) — can be array of shape
               (n_points, 6) for spatial grid or scalar (6,) for single point.
    Returns: CFS value(s) [Pa]
    """
    stress_6d = np.asarray(stress_6d)
    scalar_input = (stress_6d.ndim == 1)

    if scalar_input:
        stress_6d = stress_6d.reshape(1, -1)

    n_recv, slip_recv = receiver_geometry(receiver_strike, receiver_dip,
                                           receiver_rake)
    cfs = np.zeros(stress_6d.shape[0])

    for i in range(stress_6d.shape[0]):
        sxx, syy, szz, sxy, syz, szx = stress_6d[i]
        sigma = np.array([[sxx, sxy, szx],
                          [sxy, syy, syz],
                          [szx, syz, szz]])
        tau = n_recv @ sigma @ slip_recv
        sigma_n = n_recv @ sigma @ n_recv
        cfs[i] = tau + mu_friction * sigma_n

    return cfs[0] if scalar_input else cfs


def linear_superposition(basis_stresses, coeffs):
    """Linearly combine basis stress fields by coefficients.

    basis_stresses: list of (n_points, 6) arrays
    coeffs: (n_basis,) array of weights
    Returns: (n_points, 6) combined stress
    """
    sigma_combined = None
    for stress_i, c in zip(basis_stresses, coeffs):
        if abs(c) < 1e-6:
            continue
        if sigma_combined is None:
            sigma_combined = c * stress_i
        else:
            sigma_combined += c * stress_i
    return sigma_combined
