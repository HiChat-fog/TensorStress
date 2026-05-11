"""
Demo: Weighted Ensemble with Physical Constraints.

Synthetic data generation — no external files needed.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from tensorstress.weighted_ensemble import (
    WeightedEnsemble, make_zero_event_weights, compute_bath_offset
)

# ============================================================
# 1. Generate synthetic training data
# ============================================================
np.random.seed(42)
n_train = 500
n_test = 50
n_features = 12

X_train = np.random.randn(n_train, n_features).astype(np.float32)
X_test = np.random.randn(n_test, n_features).astype(np.float32)

# Three targets with different signal-to-noise ratios
# T1: short-term, many zeros
# T2: medium-term, fewer zeros
# T3: long-term, rare zeros
source_mag = np.random.uniform(5.0, 9.0, n_train)

Y_train = {}
Y_train['T1'] = np.maximum(0, source_mag - 1.5 + np.random.randn(n_train) * 0.5)
Y_train['T2'] = np.maximum(0, source_mag - 1.2 + np.random.randn(n_train) * 0.6)
Y_train['T3'] = np.maximum(0, source_mag - 1.8 + np.random.randn(n_train) * 0.7)

# Simulate ~30% zero-event rate for T1
zero_mask_t1 = np.random.rand(n_train) < 0.3
Y_train['T1'][zero_mask_t1] = 0.0

# ============================================================
# 2. Configure weighted training
# ============================================================
print("=" * 60)
print("WEIGHTED ENSEMBLE DEMO")
print("=" * 60)
print(f"Training: {n_train} samples, {n_features} features")
print(f"Test:     {n_test} samples")
print(f"T1 zero-rate: {(Y_train['T1'] < 0.5).mean():.1%}")
print(f"T2 zero-rate: {(Y_train['T2'] < 0.5).mean():.1%}")
print()

# Weight schedule: down-weight zero events but keep them
W = make_zero_event_weights(Y_train, {
    'T1': {'zero_weight': 0.05, 'threshold': 0.5},
    'T2': {'zero_weight': 0.12, 'threshold': 0.5},
    'T3': {'zero_weight': 1.0,  'threshold': 0.5},
})

for t in ['T1', 'T2', 'T3']:
    eff = (W[t] > 0.5).sum()
    print(f"  {t}: {eff}/{n_train} effective samples "
          f"(weighted: {W[t].sum():.1f})")

# ============================================================
# 3. Train ensemble
# ============================================================
X_dict = {'T1': X_train, 'T2': X_train, 'T3': X_train}
X_test_dict = {'T1': X_test, 'T2': X_test, 'T3': X_test}

print("\nTraining 7-seed ensemble...")
ensemble = WeightedEnsemble(n_seeds=7)
ensemble.fit(X_dict, Y_train, W)

# Predict
preds = ensemble.predict(X_test_dict)
preds_unc = ensemble.predict_with_uncertainty(X_test_dict)

print(f"\nPredictions (first 5 test samples):")
print(f"{'Sample':>8} {'T1':>8} {'T2':>8} {'T3':>8}")
print("-" * 36)
for i in range(5):
    print(f"{i:8d} {preds['T1'][i]:8.3f} {preds['T2'][i]:8.3f} "
          f"{preds['T3'][i]:8.3f}")

# ============================================================
# 4. Physical constraint: Bath's law
# ============================================================
bath_offset = compute_bath_offset(Y_train['T1'], source_mag)
print(f"\nBath's law offset (from training data): {bath_offset:.2f}")
print(f"(Applied post-hoc: max(T1_pred, source_mag - bath_offset))")

print("\nEnsemble demo complete. No external data required.")
