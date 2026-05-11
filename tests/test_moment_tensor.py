"""Tests for moment tensor and CFS modules."""

import numpy as np
import pytest
from tensorstress.moment_tensor import (
    mt_from_sdr, receiver_geometry, basis_mt_decomposition, mt_to_flat
)
from tensorstress.cfs import (
    analytical_stress_field, compute_cfs, linear_superposition
)


class TestMomentTensor:
    def test_mt_shape(self):
        mt = mt_from_sdr(229, 33, 138)
        assert mt.shape == (3, 3)

    def test_mt_symmetry(self):
        mt = mt_from_sdr(45, 60, -90)
        assert np.allclose(mt, mt.T)

    def test_mt_trace_zero(self):
        for s, d, r in [(0, 90, 0), (45, 30, 90), (180, 60, -90)]:
            mt = mt_from_sdr(s, d, r)
            assert abs(np.trace(mt)) < 1e-10, f"Failed for SDR=({s},{d},{r})"

    def test_receiver_geometry_unit_vectors(self):
        n, slip = receiver_geometry(0, 90, 0)
        assert abs(np.linalg.norm(n) - 1.0) < 1e-10
        assert abs(np.linalg.norm(slip) - 1.0) < 1e-10

    def test_receiver_orthogonality(self):
        n, slip = receiver_geometry(229, 33, 138)
        dot = abs(np.dot(n, slip))
        assert dot < 0.1, f"n and slip should be near-orthogonal, got dot={dot}"

    def test_basis_decomposition_reconstruction(self):
        s, d, r = 229, 33, 138
        coeffs = basis_mt_decomposition(s, d, r)
        target = mt_to_flat(mt_from_sdr(s, d, r))
        from tensorstress.moment_tensor import BASIS_MT
        reconstructed = BASIS_MT.T @ coeffs
        assert np.allclose(target, reconstructed, atol=1e-10)

    def test_mt_to_flat(self):
        mt = mt_from_sdr(0, 90, 0)
        flat = mt_to_flat(mt)
        assert flat.shape == (6,)
        assert flat[0] == mt[0, 0]
        assert flat[3] == mt[0, 1]


class TestCFS:
    def test_analytical_cfs_returns_all_keys(self):
        result = analytical_stress_field(7.9, 14, 229, 33, 138)
        expected = {'cfs_thrust', 'cfs_normal', 'cfs_strike',
                     'cfs_oblique1', 'cfs_oblique2', 'cfs_max'}
        assert set(result.keys()) == expected

    def test_analytical_cfs_max_is_max(self):
        result = analytical_stress_field(7.0, 10, 0, 90, 0)
        others = [result[k] for k in result if k != 'cfs_max']
        assert abs(result['cfs_max'] - max(others)) < 1e-6

    def test_compute_cfs_scalar(self):
        sigma = np.array([1.0, 0.5, -0.3, 0.2, 0.1, -0.1])
        cfs = compute_cfs(sigma, 0, 90, 0)
        assert np.ndim(cfs) == 0

    def test_compute_cfs_grid(self):
        sigma_grid = np.random.randn(100, 6)
        cfs = compute_cfs(sigma_grid, 0, 90, 0)
        assert cfs.shape == (100,)

    def test_linear_superposition(self):
        basis = [np.ones((10, 6)) * i for i in range(5)]
        coeffs = np.array([1, 0, 0, 0, 0])
        result = linear_superposition(basis, coeffs)
        assert np.allclose(result, basis[0])
