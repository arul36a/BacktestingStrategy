"""
Machine learning ensemble for directional / return forecasting with leakage controls.

Supports XGBoost, LightGBM, RandomForest, optional LSTM via TensorFlow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Literal, Mapping, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:  # pragma: no cover
    XGBClassifier = XGBRegressor = None  # type: ignore

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMClassifier = LGBMRegressor = None  # type: ignore

from .feature_engineering import make_supervised_frames
from .utils import project_root

_LOGGER = logging.getLogger(__name__)


ModeLiteral = Literal["classification", "regression"]


@dataclass
class MLArtifacts:
    model: Any
    feature_columns: List[str]
    mode: ModeLiteral
    regime_models: dict[int, Any] = field(default_factory=dict)
    use_regime_models: bool = False


def purge_split_indices(n: int, train_end: int, purge: int, test_end: int) -> tuple[slice, slice]:
    train_slice = slice(0, train_end - purge)
    test_slice = slice(train_end, test_end)
    return train_slice, test_slice


def _build_base_estimators(
    mode: ModeLiteral,
    seed: int,
    include_rf: bool = True,
    include_xgb: bool = True,
    include_lgb: bool = True,
) -> list[tuple[str, Any]]:
    ests: list[tuple[str, Any]] = []
    if mode == "classification":
        if include_rf:
            ests.append(("rf", RandomForestClassifier(random_state=seed, class_weight="balanced_subsample")))
        if include_xgb and XGBClassifier:
            ests.append(
                (
                    "xgb",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.03,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        eval_metric="logloss",
                        random_state=seed,
                        tree_method="hist",
                        n_jobs=-1,
                        verbosity=0,
                    ),
                )
            )
        if include_lgb and LGBMClassifier:
            ests.append(
                (
                    "lgb",
                    LGBMClassifier(
                        n_estimators=400,
                        learning_rate=0.03,
                        num_leaves=48,
                        random_state=seed,
                        verbosity=-1,
                        n_jobs=-1,
                        class_weight="balanced",
                    ),
                )
            )
    else:
        if include_rf:
            ests.append(("rf", RandomForestRegressor(random_state=seed)))
        if include_xgb and XGBRegressor:
            ests.append(
                (
                    "xgb",
                    XGBRegressor(
                        n_estimators=400,
                        max_depth=6,
                        learning_rate=0.03,
                        subsample=0.8,
                        random_state=seed,
                        tree_method="hist",
                        n_jobs=-1,
                        verbosity=0,
                    ),
                )
            )
        if include_lgb and LGBMRegressor:
            ests.append(
                (
                    "lgb",
                    LGBMRegressor(
                        n_estimators=500,
                        learning_rate=0.03,
                        num_leaves=48,
                        random_state=seed,
                        verbosity=-1,
                        n_jobs=-1,
                    ),
                )
            )

    return ests


class GoldMLEngine:
    """Training, tuning, persistence, inference for GOLD strategy features."""

    def __init__(self, config: dict, seed: int = 42) -> None:
        self.cfg = config
        self.seed = seed
        self.ml_cfg = dict(config.get("ml") or {})

    def supervised_dataset(
        self,
        feats: pd.DataFrame,
        close: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Returns X, y, forward_returns aligned (forward_returns mainly for diagnostics)."""
        hz = int(self.ml_cfg.get("forward_horizon", 1))
        mode_mode = self.ml_cfg.get("target_mode", "classification")  # type: ignore[arg-type]
        thresh = float(self.ml_cfg.get("classification_threshold", 0.0))
        label_mode = str(self.ml_cfg.get("label_mode", "forward_return"))
        tb_cfg = dict(self.ml_cfg.get("triple_barrier") or {})
        fr = close.shift(-hz) / close - 1.0
        X, y = make_supervised_frames(
            feats,
            close,
            horizon=hz,
            mode=str(mode_mode),
            classification_threshold=thresh,
            label_mode=label_mode,
            atr_series=feats.get("atr"),
            tb_tp_atr_mult=float(tb_cfg.get("tp_atr_mult", 1.5)),
            tb_sl_atr_mult=float(tb_cfg.get("sl_atr_mult", 1.0)),
            tb_min_abs_ret=float(tb_cfg.get("min_abs_ret", 0.0)),
            drop_na=True,
        )
        fwd = fr.reindex(X.index)
        return X, y, fwd

    def fit_regime_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        regimes: pd.Series,
        *,
        mode: ModeLiteral,
        tune: bool = False,
    ) -> dict[int, Any]:
        models: dict[int, Any] = {}
        for regime in sorted(pd.Series(regimes.dropna().unique()).astype(int).tolist()):
            mask = regimes.reindex(X.index) == regime
            Xr = X.loc[mask]
            yr = y.loc[mask]
            if len(Xr) < max(100, int(0.08 * len(X))):
                continue
            models[int(regime)] = self.fit_ensemble(Xr, yr, mode=mode, tune=tune)
        return models

    def fit_ensemble(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        mode: ModeLiteral,
        tune: bool = True,
        sample_weight: Optional[np.ndarray] = None,
    ) -> VotingClassifier | VotingRegressor:
        seed = self.seed
        estimators = _build_base_estimators(mode, seed)
        if not estimators:
            raise RuntimeError(
                "No sklearn estimators available — ensure scikit-learn and optional xgboost/lightgbm are installed.",
            )
        EnsembleType = VotingClassifier if mode == "classification" else VotingRegressor
        tuning = dict(self.ml_cfg.get("tuning") or {})

        if tune and tuning.get("n_iter") and mode == "classification" and dict(estimators).get("rf") is not None:
            param_dist = {"n_estimators": [150, 300, 450], "max_depth": [4, 6, None]}
            searcher = RandomizedSearchCV(
                RandomForestClassifier(random_state=seed, class_weight="balanced_subsample"),
                param_distributions=param_dist,
                n_iter=min(int(tuning["n_iter"]), 9),
                cv=max(2, int(tuning.get("cv_splits", 3))),
                random_state=seed,
                scoring="accuracy",
                n_jobs=-1,
                verbose=0,
            )
            searcher.fit(X.values, y.values)
            rf_best = searcher.best_estimator_
            swapped = [(nam, rf_best if nam == "rf" else est) for nam, est in estimators]
            ensemble = EnsembleType(estimators=list(swapped), n_jobs=-1)
            ensemble.fit(X.values, y.values, sample_weight=sample_weight)  # type: ignore[arg-type]
            _LOGGER.info("Fitted Voting ensemble with tuned RF (%s)", mode)
            return ensemble

        ensemble = EnsembleType(estimators=list(estimators), n_jobs=-1)
        ensemble.fit(X.values, y.values, sample_weight=sample_weight)  # type: ignore[arg-type]
        _LOGGER.info("Fitted baseline Voting ensemble (%s)", mode)
        return ensemble

    def walk_forward_evaluation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        mode: ModeLiteral,
        n_splits: int,
        purge_gap: int,
    ) -> pd.DataFrame:
        """
        Expanding-history walk-forward evaluation.

        Boundaries chop the sorted sample into chronological test folds. Training always
        spans `[0 .. test_start_idx - purge_gap)` so overlapping label windows adjacent
        to the test horizon are withheld.
        """
        xs = X.sort_index()
        ys = y.reindex(xs.index)
        n = int(len(xs))
        if n < 250:
            _LOGGER.warning("walk_forward_evaluation: thin sample (n=%s)", n)

        boundaries = sorted(
            {int(round(max(96, min(n - 48, int(n * a))))) for a in np.linspace(0.4, 0.995, max(5, int(n_splits) + 3))},
        )

        rows: list[dict[str, Any]] = []
        for idx in range(1, len(boundaries) - 1):
            test_start = boundaries[idx]
            test_end = boundaries[idx + 1]
            if test_end <= test_start + 8:
                continue

            train_exclusive_end = max(96, test_start - purge_gap)
            X_tr = xs.iloc[:train_exclusive_end]
            y_tr = ys.iloc[:train_exclusive_end]

            engine = GoldMLEngine(self.cfg, seed=self.seed)
            model = engine.fit_ensemble(X_tr, y_tr, mode=mode, tune=False)

            X_te = xs.iloc[test_start:test_end]
            y_te = ys.iloc[test_start:test_end]
            preds = model.predict(X_te.values)

            row: dict[str, Any]
            if mode == "classification":
                try:
                    pr = model.predict_proba(X_te.values)[:, 1]
                    auc = float(roc_auc_score(y_te, pr))
                except Exception:
                    auc = float("nan")
                row = {
                    "test_period": (str(xs.index[test_start]), str(xs.index[min(test_end - 1, n - 1)])),
                    "train_rows": len(X_tr),
                    "accuracy": accuracy_score(y_te, preds),
                    "auc": auc,
                    "samples": len(y_te),
                }
            else:
                rmse_val = float(np.sqrt(mean_squared_error(y_te, preds)))
                row = {
                    "test_period": (str(xs.index[test_start]), str(xs.index[min(test_end - 1, n - 1)])),
                    "train_rows": len(X_tr),
                    "rmse": rmse_val,
                    "samples": len(y_te),
                }

            rows.append(row)

        return pd.DataFrame(rows)

    def save(
        self,
        ensemble: Any,
        cols: Sequence[str],
        rel_path: str,
        *,
        regime_models: Optional[dict[int, Any]] = None,
        use_regime_models: bool = False,
    ) -> Path:
        root = project_root()
        raw_path = Path(rel_path).expanduser()
        path = raw_path.resolve() if raw_path.is_absolute() else (root / raw_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = MLArtifacts(
            model=ensemble,
            feature_columns=list(cols),
            mode=self.ml_cfg.get("target_mode", "classification"),  # type: ignore[arg-type]
            regime_models=regime_models or {},
            use_regime_models=use_regime_models,
        )
        joblib.dump(payload, path)
        symlink = root / "models/latest_ensemble.pkl"
        try:
            if symlink.exists() or symlink.is_symlink():
                symlink.unlink()
            symlink.symlink_to(path)
        except OSError:
            joblib.dump(payload, symlink)
        _LOGGER.info("Saved ML artifacts → %s", path)
        return path

    @staticmethod
    def load(rel_path: str) -> MLArtifacts:
        raw_path = Path(rel_path).expanduser()
        path = raw_path.resolve() if raw_path.is_absolute() else (project_root() / raw_path).resolve()
        obj = joblib.load(path)
        if isinstance(obj, MLArtifacts):
            return obj
        raise ValueError("Invalid artifact format")


class LSTMModule:
    """Optional GPU-capable recurrent model with TensorFlow; isolated to avoid TF import penalties."""

    def __init__(self, seq_len: int = 32, lstm_units: int = 48) -> None:
        self.seq_len = seq_len
        self.units = lstm_units

    def build_and_fit(self, features: pd.DataFrame, y: pd.Series, epochs: int = 15, batch_size: int = 64) -> Any:
        try:
            import tensorflow as tf
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("TensorFlow not installed — omit LSTMModule or pip install tensorflow") from exc

        x = features.values.astype("float32")
        yy = y.values.astype("float32")
        nx = []
        ny = []
        for i in range(self.seq_len, len(x)):
            nx.append(x[i - self.seq_len : i])
            ny.append(yy[i])

        xt = tf.convert_to_tensor(np.array(nx), dtype=tf.float32)
        yt = tf.convert_to_tensor(np.array(ny).reshape(-1, 1), dtype=tf.float32)

        inp = tf.keras.Input(shape=(self.seq_len, x.shape[1]))
        lstm_hidden = tf.keras.layers.LSTM(self.units)(inp)
        out = tf.keras.layers.Dense(1, activation="sigmoid")(lstm_hidden)

        model = tf.keras.Model(inputs=inp, outputs=out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.BinaryCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

        callbacks = []
        callbacks.append(tf.keras.callbacks.EarlyStopping(patience=3))
        history = model.fit(xt, yt, epochs=int(epochs), batch_size=batch_size, validation_split=0.15, callbacks=callbacks, verbose=0)
        setattr(model, "history_", history)
        setattr(model, "feature_cols_", features.columns.to_list())
        return model


def feature_importances_from_rf_submodel(
    ensemble: Any,
    feature_cols: Sequence[str],
) -> Optional[pd.Series]:
    """Inspect voting ensemble for sklearn RF subtree importances."""
    cols = np.array(feature_cols).astype(str)
    try:
        for name, est in ensemble.named_estimators_.items():  # type: ignore[attr-defined]
            if name == "rf" and hasattr(est, "feature_importances_"):
                weights = getattr(est.feature_importances_, "shape", ())
                idx = getattr(ensemble, "feature_names_in_", cols)
                if len(idx) != len(est.feature_importances_):
                    idx = cols[: len(est.feature_importances_)]
                return pd.Series(est.feature_importances_, index=idx)
    except AttributeError:
        pass
    return None


def predict_probabilities(
    artifact: MLArtifacts,
    X: pd.DataFrame,
    regimes: Optional[pd.Series] = None,
) -> pd.Series:
    block = X.reindex(columns=artifact.feature_columns).astype(float).fillna(0)
    if not artifact.use_regime_models or not artifact.regime_models or regimes is None:
        if hasattr(artifact.model, "predict_proba"):
            out = artifact.model.predict_proba(block.values)[:, 1]
        else:
            out = artifact.model.predict(block.values).astype(float)
        return pd.Series(out, index=block.index, name="ml_prob")

    global_pred = (
        artifact.model.predict_proba(block.values)[:, 1]
        if hasattr(artifact.model, "predict_proba")
        else artifact.model.predict(block.values).astype(float)
    )
    pred = pd.Series(global_pred, index=block.index, dtype=float)
    reg = regimes.reindex(block.index)
    for regime, model in artifact.regime_models.items():
        mask = reg == regime
        if not mask.any():
            continue
        sub = block.loc[mask]
        if hasattr(model, "predict_proba"):
            pred.loc[mask] = model.predict_proba(sub.values)[:, 1]
        else:
            pred.loc[mask] = model.predict(sub.values).astype(float)
    return pred.rename("ml_prob")
