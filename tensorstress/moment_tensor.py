"""
Moment Tensor Mathematics for Source Mechanics.

Core operations:
1. Strike/Dip/Rake → normalized moment tensor (double-couple)
2. Receiver fault geometry (normal vector + slip direction)
3. Basis moment tensor decomposition for linear superposition

All formulas are standard in computational geomechanics;
the implementation emphasizes numerical stability and vectorization.
"""

import numpy as np


def mt_from_sdr(strike, dip, rake):
    """Convert fault geometry to normalized moment tensor (double-couple).

    strike, dip, rake: angles in degrees
    Returns: 3x3 symmetric moment tensor (normalized)
    """
    s, d, r = np.radians(strike), np.radians(dip), np.radians(rake)

    Mxx = -(np.sin(d) * np.cos(r) * np.sin(2 * s)
            + np.sin(2 * d) * np.sin(r) * np.sin(s) ** 2)
    Myy = (np.sin(d) * np.cos(r) * np.sin(2 * s)
           - np.sin(2 * d) * np.sin(r) * np.cos(s) ** 2)
    Mzz = np.sin(2 * d) * np.sin(r)
    Mxy = (np.sin(d) * np.cos(r) * np.cos(2 * s)
           + 0.5 * np.sin(2 * d) * np.sin(r) * np.sin(2 * s))
    Mxz = -(np.cos(d) * np.cos(r) * np.cos(s)
            + np.cos(2 * d) * np.sin(r) * np.sin(s))
    Myz = -(np.cos(d) * np.cos(r) * np.sin(s)
            - np.cos(2 * d) * np.sin(r) * np.cos(s))

    return np.array([[Mxx, Mxy, Mxz],
                     [Mxy, Myy, Myz],
                     [Mxz, Myz, Mzz]])


def mt_to_flat(mt):
    """Flatten symmetric 3x3 MT to 6-component vector (Voigt notation)."""
    return np.array([mt[0, 0], mt[1, 1], mt[2, 2],
                     mt[0, 1], mt[0, 2], mt[1, 2]])


def receiver_geometry(strike, dip, rake):
    """Compute receiver fault normal vector and slip direction.

    Returns: (normal_vec, slip_vec) each as (3,) np.array
    """
    s, d, r = np.radians(strike), np.radians(dip), np.radians(rake)

    n = np.array([-np.sin(d) * np.sin(s),
                   np.sin(d) * np.cos(s),
                   np.cos(d)])

    slip = np.array([
        np.cos(r) * np.cos(s) + np.sin(r) * np.cos(d) * np.sin(s),
        np.cos(r) * np.sin(s) - np.sin(r) * np.cos(d) * np.cos(s),
        np.sin(r) * np.sin(d),
    ])

    return n, slip


# Standard basis moment tensors for double-couple decomposition
BASIS_SDR = [
    (315, 90, 0),      # Strike-slip
    (0, 90, 0),         # Strike-slip (orthogonal)
    (0, 0, 180),       # Vertical dip-slip
    (180, 45, -90),     # Normal fault
    (0, 0, 90),        # Thrust fault
]

BASIS_MT = np.array([mt_to_flat(mt_from_sdr(*sdr)) for sdr in BASIS_SDR])


def basis_mt_decomposition(strike, dip, rake):
    """Decompose arbitrary double-couple MT into basis coefficients.

    Any double-couple moment tensor = sum(coeff_i * BASIS_MT_i).
    Coefficients solved via least squares.

    Returns: coeffs (5,) np.array
    """
    target = mt_to_flat(mt_from_sdr(strike, dip, rake))
    coeffs = np.linalg.lstsq(BASIS_MT.T, target, rcond=None)[0]
    return coeffs


def cfs_from_stress_tensor(sigma_3x3, receiver_strike, receiver_dip,
                           receiver_rake, mu_friction=0.4):
    """Compute Coulomb Failure Stress from a 3x3 stress tensor.

    CFS = tau + mu * sigma_n
    where tau = shear stress on receiver plane,
          sigma_n = normal stress on receiver plane.

    Returns: scalar CFS value
    """
    n, slip = receiver_geometry(receiver_strike, receiver_dip, receiver_rake)
    tau = n @ sigma_3x3 @ slip
    sigma_n = n @ sigma_3x3 @ n
    return tau + mu_friction * sigma_n
