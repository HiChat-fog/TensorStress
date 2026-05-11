"""
Comprehensive ML Model Audit Framework.

6-dimension audit for regression models:
1. Cross-validation stability (5-fold CV MAE)
2. Feature importance analysis (no single feature > 20%)
3. Prediction distribution check (no anomaly in min/max/mean)
4. Edge-case inspection (extreme values in input space)
5. Physical baseline comparison (model vs known scaling law)
6. Ablation verification (with vs without key components)
"""

import numpy as np
import statistics
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


class ModelAuditor:
    """Systematic audit for regression models."""

    def __init__(self, model_class=XGBRegressor, model_params=None):
        self.model_class = model_class
        self.model_params = model_params or {
            'n_estimators': 300, 'max_depth': 5,
            'learning_rate': 0.03, 'subsample': 0.8,
            'colsample_bytree': 0.7, 'random_state': 42, 'n_jobs': -1,
        }
        self.results = {}

    # ------------------------------------------------------------------
    # Audit 1: Cross-validation
    # ------------------------------------------------------------------
    def audit_cv(self, X, y, sample_weights=None, n_folds=5, target_name=''):
        """5-fold CV with and without sample weights."""
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        # With weights
        cv_weighted = []
        for tr_idx, va_idx in kf.split(X):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            w_tr = sample_weights[tr_idx] if sample_weights is not None else None

            m = self.model_class(**self.model_params)
            m.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
            cv_weighted.append(mean_absolute_error(y_va, m.predict(X_va)))

        # Without weights (baseline)
        cv_unweighted = []
        for tr_idx, va_idx in kf.split(X):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            m = self.model_class(**self.model_params)
            m.fit(X_tr, y_tr, verbose=False)
            cv_unweighted.append(mean_absolute_error(y_va, m.predict(X_va)))

        result = {
            'weighted_mae': statistics.mean(cv_weighted),
            'weighted_std': statistics.stdev(cv_weighted),
            'unweighted_mae': statistics.mean(cv_unweighted),
            'unweighted_std': statistics.stdev(cv_unweighted),
        }
        self.results[f'cv_{target_name}'] = result
        return result

    # ------------------------------------------------------------------
    # Audit 2: Feature importance
    # ------------------------------------------------------------------
    def audit_feature_importance(self, X, y, feature_names, sample_weights=None,
                                  target_name='', dominance_threshold=0.2):
        """Check for suspicious single-feature dominance."""
        m = self.model_class(**self.model_params)
        m.fit(X, y, sample_weight=sample_weights, verbose=False)

        importance = sorted(
            zip(feature_names, m.feature_importances_),
            key=lambda x: -x[1]
        )
        top3_sum = sum(v for _, v in importance[:3])

        suspicious = [name for name, val in importance if val > dominance_threshold]

        result = {
            'top3_importance_sum': top3_sum,
            'suspicious_features': suspicious,
            'top10': importance[:10],
        }
        self.results[f'importance_{target_name}'] = result
        return result

    # ------------------------------------------------------------------
    # Audit 3: Prediction distribution
    # ------------------------------------------------------------------
    def audit_distribution(self, X_train, y_train, X_test,
                           sample_weights=None, target_name=''):
        """Check prediction distribution vs training distribution."""
        m = self.model_class(**self.model_params)
        m.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

        train_pred = m.predict(X_train)
        test_pred = m.predict(X_test)

        warnings = []
        if test_pred.min() < -0.5:
            warnings.append(f'Negative predictions (min={test_pred.min():.2f})')
        if test_pred.max() > np.max(y_train) * 1.5:
            warnings.append(f'Extrapolation (max={test_pred.max():.2f} vs '
                            f'train max={np.max(y_train):.2f})')

        result = {
            'train_pred': {'min': float(train_pred.min()),
                           'max': float(train_pred.max()),
                           'mean': float(train_pred.mean())},
            'train_true': {'min': float(y_train.min()),
                           'max': float(y_train.max()),
                           'mean': float(y_train.mean())},
            'test_pred': {'min': float(test_pred.min()),
                          'max': float(test_pred.max()),
                          'mean': float(test_pred.mean())},
            'train_test_diff': float(abs(train_pred.mean() - test_pred.mean())),
            'warnings': warnings,
        }
        self.results[f'distribution_{target_name}'] = result
        return result

    # ------------------------------------------------------------------
    # Audit 4: Edge cases
    # ------------------------------------------------------------------
    def audit_edge_cases(self, X, y, edge_indices, sample_weights=None,
                          target_name=''):
        """Inspect predictions on extreme input values."""
        m = self.model_class(**self.model_params)
        m.fit(X, y, sample_weight=sample_weights, verbose=False)

        edge_results = []
        for idx in edge_indices:
            x = X[idx:idx + 1]
            pred = float(m.predict(x)[0])
            edge_results.append({
                'index': idx, 'prediction': pred, 'true': float(y[idx]),
            })

        result = {'edge_cases': edge_results}
        self.results[f'edge_{target_name}'] = result
        return result

    # ------------------------------------------------------------------
    # Audit 5: Physical baseline
    # ------------------------------------------------------------------
    def audit_physical_baseline(self, baseline_predictions, model_predictions,
                                 true_values, target_name=''):
        """Compare model against a simple physical baseline."""
        baseline_mae = statistics.mean([
            abs(p - t) for p, t in zip(baseline_predictions, true_values)
        ])
        model_mae = statistics.mean([
            abs(p - t) for p, t in zip(model_predictions, true_values)
        ])

        improvement = (baseline_mae - model_mae) / baseline_mae * 100

        result = {
            'baseline_mae': baseline_mae,
            'model_mae': model_mae,
            'improvement_percent': improvement,
        }
        self.results[f'baseline_{target_name}'] = result
        return result

    # ------------------------------------------------------------------
    # Audit 6: Ablation
    # ------------------------------------------------------------------
    def audit_ablation(self, X, y, ablated_X, sample_weights=None,
                        target_name=''):
        """Compare performance with vs without key feature groups."""
        # Full model
        m_full = self.model_class(**self.model_params)
        m_full.fit(X, y, sample_weight=sample_weights, verbose=False)
        full_pred = m_full.predict(X)

        # Ablated model (reduced features)
        m_ablated = self.model_class(**self.model_params)
        m_ablated.fit(ablated_X, y, sample_weight=sample_weights, verbose=False)
        ablated_pred = m_ablated.predict(ablated_X)

        result = {
            'full_mae': float(mean_absolute_error(y, full_pred)),
            'ablated_mae': float(mean_absolute_error(y, ablated_pred)),
            'degradation': float(mean_absolute_error(y, ablated_pred)
                                 - mean_absolute_error(y, full_pred)),
        }
        self.results[f'ablation_{target_name}'] = result
        return result

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self):
        """Print all audit results."""
        print("=" * 60)
        print("MODEL AUDIT SUMMARY")
        print("=" * 60)
        for key, val in sorted(self.results.items()):
            print(f"\n--- {key} ---")
            if isinstance(val, dict):
                for k, v in val.items():
                    print(f"  {k}: {v}")
            else:
                print(f"  {val}")
        return self.results
