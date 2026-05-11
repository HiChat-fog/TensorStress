"""
Demo: Moment Tensor Mathematics & Coulomb Failure Stress.

Pure computation — no external data required.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from tensorstress.moment_tensor import (
    mt_from_sdr, receiver_geometry, basis_mt_decomposition
)
from tensorstress.cfs import analytical_stress_field, compute_cfs

# ============================================================
# 1. Strike/Dip/Rake → Moment Tensor
# ============================================================
print("=" * 60)
print("1. MOMENT TENSOR FROM FAULT GEOMETRY")
print("=" * 60)

# Example: thrust fault (strike=229, dip=33, rake=138)
s, d, r = 229, 33, 138
mt = mt_from_sdr(s, d, r)
print(f"Fault: strike={s} deg, dip={d} deg, rake={r} deg")
print(f"Moment Tensor (normalized):")
print(f"  [[{mt[0,0]:.4f}  {mt[0,1]:.4f}  {mt[0,2]:.4f}]")
print(f"   [{mt[1,0]:.4f}  {mt[1,1]:.4f}  {mt[1,2]:.4f}]")
print(f"   [{mt[2,0]:.4f}  {mt[2,1]:.4f}  {mt[2,2]:.4f}]]")

# Verify: trace should be ~0 (double-couple)
trace = np.trace(mt)
print(f"Trace = {trace:.6f} (should be ~0 for double-couple)")

# ============================================================
# 2. Receiver Fault Geometry
# ============================================================
print()
print("=" * 60)
print("2. RECEIVER FAULT GEOMETRY")
print("=" * 60)

n, slip = receiver_geometry(0, 90, 0)  # vertical strike-slip
print(f"Receiver (strike=0, dip=90, rake=0):")
print(f"  Normal vector: ({n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f})")
print(f"  Slip vector:   ({slip[0]:.4f}, {slip[1]:.4f}, {slip[2]:.4f})")

# ============================================================
# 3. Basis Decomposition
# ============================================================
print()
print("=" * 60)
print("3. BASIS MT DECOMPOSITION")
print("=" * 60)

coeffs = basis_mt_decomposition(s, d, r)
print(f"Thrust event decomposed into 5 basis coefficients:")
for i, (basis_sdr, c) in enumerate(zip(
    [(315,90,0), (0,90,0), (0,0,180), (180,45,-90), (0,0,90)], coeffs
)):
    bar = '#' * int(abs(c) * 20) if abs(c) > 0.01 else ''
    print(f"  Basis {i} ({basis_sdr[0]:3d},{basis_sdr[1]:2d},{basis_sdr[2]:4d}): "
          f"{c:+.4f} {bar}")

# ============================================================
# 4. Analytical CFS
# ============================================================
print()
print("=" * 60)
print("4. ANALYTICAL COULOMB FAILURE STRESS")
print("=" * 60)

# M7.9 at 14km depth
results = analytical_stress_field(magnitude=7.9, depth_km=14,
                                   strike=229, dip=33, rake=138)
print("Source: M7.9, depth=14km, thrust mechanism")
print("CFS on canonical receiver types (MPa):")
for k, v in sorted(results.items()):
    print(f"  {k:20s} = {v:+.4f}")
print(f"\n  CFS_max = {results['cfs_max']:+.4f} MPa")

# ============================================================
# 5. CFS from Arbitrary Stress Tensor
# ============================================================
print()
print("=" * 60)
print("5. CFS FROM STRESS TENSOR")
print("=" * 60)

# Synthetic stress tensor (MPa)
sigma = np.array([1.0, 0.5, -0.3, 0.2, 0.1, -0.1])  # Sxx,Syy,Szz,Sxy,Syz,Szx
cfs_val = compute_cfs(sigma, receiver_strike=0, receiver_dip=90,
                       receiver_rake=0)
print(f"Stress tensor (Voigt): {sigma}")
print(f"Receiver: vertical strike-slip")
print(f"CFS = {cfs_val:.4f} Pa")

print()
print("All demos passed. No data files required.")
