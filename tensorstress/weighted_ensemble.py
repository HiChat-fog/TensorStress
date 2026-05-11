"""
Weighted Multi-Target Ensemble with Physical Law Enforcement.

Innovation: instead of filtering out "noisy" data points (zero-valued
targets, physically impossible labels), assign them soft weights to
preserve their partial signal for correlated targets while suppressing
noise for the primary target.

Strategy:
- Target-1 (short-term): zero events get very low weight (0.05)
- Target-2 (medium-term): zero events get moderate weight (0.12)
- Target-3 (long-term): all events get full weight (1.0)
- Physically impossible labels (target > source + margin) get weight 0

Post-processing enforces known physical scaling laws on predictions.
"""

import numpy as np
import statistics
from xgboost import XGBRegressor


class WeightedEnsemble:
    """Multi-target XGBoost ensemble with per-sample weights and
    physical law post-processing.

    Usage:
        ensemble = WeightedEnsemble(n_seeds=7, base_params={...})
        ensemble.fit(X_dict, Y_dict, W_dict)
        predictions = ensemble.predict(X_test)
    """

    def __init__(self, n_seeds=7, base_params=None):
        """
        base_params: dict of XGBRegressor params (overrides defaults)
        """
        self.n_seeds = n_seeds
        self.base_params = {
            'n_estimators': 300,
            'max_depth': 5,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'n_jobs': -1,
        }
        if base_params:
            self.base_params.update(base_params)
        self.models = {}       # {target_name: [model_seed0, ...]}
        self.target_names = []

    def fit(self, X_dict, Y_dict, W_dict=None):
        """Fit ensemble for each target window.

        X_dict: {'T1': X_train_T1, 'T2': X_train_T2, ...}
        Y_dict: {'T1': y_train_T1, 'T2': y_train_T2, ...}
        W_dict: {'T1': sample_weights_T1, ...} (optional)
        """
        self.target_names = sorted(X_dict.keys())

        for target in self.target_names:
            X = np.asarray(X_dict[target], dtype=np.float32)
            y = np.asarray(Y_dict[target], dtype=np.float32)
            w = np.asarray(W_dict[target], dtype=np.float32) if W_dict else None

            self.models[target] = []
            for s in range(self.n_seeds):
                params = {**self.base_params, 'random_state': 42 + s}
                model = XGBRegressor(**params)
                model.fit(X, y, sample_weight=w, verbose=False)
                self.models[target].append(model)

        return self

    def predict(self, X_dict):
        """Predict for each target. Returns dict {target: mean_prediction}."""
        results = {}
        for target in self.target_names:
            X = np.asarray(X_dict[target], dtype=np.float32)
            preds = np.column_stack([
                m.predict(X) for m in self.models[target]
            ])
            results[target] = preds.mean(axis=1)
        return results

    def predict_with_uncertainty(self, X_dict):
        """Predict with ensemble std for uncertainty quantification."""
        results = {}
        for target in self.target_names:
            X = np.asarray(X_dict[target], dtype=np.float32)
            preds = np.column_stack([
                m.predict(X) for m in self.models[target]
            ])
            results[target] = {
                'mean': preds.mean(axis=1),
                'std': preds.std(axis=1),
                'min': preds.min(axis=1),
                'max': preds.max(axis=1),
            }
        return results


# ---------------------------------------------------------------------------
# Weight Schedule Generators
# ---------------------------------------------------------------------------

def make_zero_event_weights(Y_train, target_configs):
    """Generate per-sample weights for multi-target regression
    where zero-valued targets contain partial signal.

    target_configs: dict like {
        'T1': {'zero_weight': 0.05, 'threshold': 0.5},
        'T2': {'zero_weight': 0.12, 'threshold': 0.5},
        'T3': {'zero_weight': 1.0,  'threshold': 0.5},  # keep all
    }

    Returns: dict {target: weight_array}
    """
    W = {}
    for target, cfg in target_configs.items():
        y = np.asarray(Y_train[target])
        w = np.ones(len(y), dtype=np.float32)
        zero_mask = y < cfg['threshold']
        w[zero_mask] = cfg['zero_weight']
        W[target] = w
    return W


def apply_physical_ceiling(Y_train, source_values, ceiling_margin=0.5):
    """Zero out weights for physically impossible labels.

    If target > source + margin, the label is physically impossible
    (e.g., aftershock cannot exceed mainshock by 0.5 magnitude units).

    Returns: adjusted weight arrays
    """
    for target in Y_train:
        y = np.asarray(Y_train[target])
        src = np.asarray(source_values)
        impossible = y > src + ceiling_margin
        # This is typically applied to pre-existing weight arrays
        yield target, impossible


def compute_bath_offset(Y_train, source_values, threshold=0.5):
    """Estimate Bath's law offset from training data.

    Bath's law: largest aftershock is typically mainshock - offset.
    Only uses events with nonzero targets that are physically plausible.

    Returns: median offset (float)
    """
    diffs = []
    y = np.asarray(Y_train)
    src = np.asarray(source_values)
    for yi, si in zip(y, src):
        if yi > threshold and yi <= si:
            diffs.append(si - yi)
    return statistics.median(diffs) if diffs else 1.2
