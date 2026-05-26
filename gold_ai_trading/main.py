#!/usr/bin/env python3
"""
Gold AI Trading — orchestration CLI for VectorBT research, ML training,
and Backtrader validation pipelines.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from src.backtrader_engine import GoldBacktraderEngine
from src.data_loader import GoldDataLoader
from src.evaluation import monte_carlo_returns_bootstrap, regime_from_volatility
from src.feature_engineering import FeaturePipeline
from src.ml_models import GoldMLEngine, feature_importances_from_rf_submodel, predict_probabilities
from src.utils import ensure_dirs, load_app_config, project_root, resolve_path, setup_logging
from src.vectorbt_research import VectorBTGoldResearch, save_research_snapshot

_LOGGER = logging.getLogger("gold_ai.cli")


def _cfg_for_command(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_app_config(args.config)
    if args.symbol:
        cfg.setdefault("data", {})["default_symbol"] = args.symbol
    ensure_dirs(cfg)
    setup_logging(cfg)
    return cfg


def cmd_research(args: argparse.Namespace) -> None:
    cfg = _cfg_for_command(args)
    loader = GoldDataLoader(cfg)
    raw = loader.download(interval=args.interval, use_cache=args.use_cache)
    if raw.empty:
        raise SystemExit("No data returned — verify symbol/network or cache.")

    higher: dict[str, Any] = {}
    pipeline = FeaturePipeline(cfg)
    feats_full = pipeline.transform(raw, higher_timeframes=higher or None)

    research = VectorBTGoldResearch(cfg)
    close = raw["close"].reindex(feats_full.index).dropna().astype(float)

    sweep = research.param_sweep_ema_fast_slow(close, fast_grid=range(8, 30, 6), slow_grid=range(40, 120, 20))
    out_dir = str(resolve_path(cfg["paths"]["reports_dir"])) + "/research"
    fig = research.heatmap_figure(sweep)
    snapshot = save_research_snapshot(out_dir, sweep, fig)
    print(f"[research] Sweep saved → {snapshot / 'vectorbt_param_sweep.csv'}")


def cmd_train_ml(args: argparse.Namespace) -> None:
    cfg = _cfg_for_command(args)
    loader = GoldDataLoader(cfg)
    raw = loader.download(interval=args.interval, use_cache=args.use_cache)
    if raw.empty:
        raise SystemExit("No OHLC rows — aborting ML train.")

    pipeline = FeaturePipeline(cfg)
    feats = pipeline.transform(raw, higher_timeframes=None)
    join = feats.join(raw["close"]).dropna(subset=["close"])

    ml_engine = GoldMLEngine(cfg)
    mode = cfg.get("ml", {}).get("target_mode", "classification")
    X, y, _fwd = ml_engine.supervised_dataset(join.drop(columns=["close"]), join["close"])

    split_ix = max(int(len(X) * float(cfg["ml"]["train_test_pct"])), len(X) // 5 + 64)
    X_tr, y_tr = X.iloc[:split_ix], y.iloc[:split_ix]

    ensemble = ml_engine.fit_ensemble(X_tr, y_tr, mode=mode, tune=not args.no_tune)
    regime_cfg = dict(cfg.get("ml", {}).get("regime_models") or {})
    use_regime = bool(regime_cfg.get("enabled", True))
    regime_col = str(regime_cfg.get("feature_col", "vol_regime"))
    regime_models: dict[int, Any] = {}
    if use_regime and regime_col in X.columns:
        regime_models = ml_engine.fit_regime_models(
            X_tr,
            y_tr,
            X_tr[regime_col],
            mode=mode,
            tune=False,
        )
    purge = int(cfg["ml"]["purge_gap_bars"])

    wf_summary = ml_engine.walk_forward_evaluation(
        X,
        y,
        mode=mode,
        n_splits=int(cfg["ml"]["walk_forward_splits"]),
        purge_gap=purge,
    )
    reports_root = resolve_path(cfg["paths"]["reports_dir"])
    reports_root.mkdir(parents=True, exist_ok=True)
    wf_summary.to_csv(reports_root / "ml_walk_forward.csv", index=False)

    artifacts = cfg.get("artifacts") or {}
    rel_model = str(artifacts.get("ml_model_joblib", "./models/stacked_ensemble.joblib"))
    path = ml_engine.save(
        ensemble,
        list(X.columns),
        rel_model,
        regime_models=regime_models,
        use_regime_models=bool(regime_models),
    )

    fi = feature_importances_from_rf_submodel(ensemble, list(X.columns))
    if fi is not None:
        fi.sort_values(ascending=False).to_csv(reports_root / "ml_rf_feature_importances.csv")

    mc = monte_carlo_returns_bootstrap(join["close"].pct_change().dropna(), simulations=500)
    mc.describe().to_csv(reports_root / "monte_carlo_bootstrap_summary.csv")

    regimes = regime_from_volatility(
        join["close"].pct_change(),
        window=int(cfg.get("regime_detection", {}).get("vol_window", 63)),
    )
    regimes.to_csv(reports_root / "vol_regime_series.csv")

    print(f"[ml] Ensemble saved → {path}")
    print(f"[ml] Walk-forward diagnostics → {reports_root / 'ml_walk_forward.csv'}")


def cmd_validate_bt(args: argparse.Namespace) -> None:
    cfg = _cfg_for_command(args)
    loader = GoldDataLoader(cfg)

    artifact_path_str = args.model or str(project_root() / "models/latest_ensemble.pkl")
    artifact = GoldMLEngine.load(artifact_path_str)

    raw = loader.download(interval=args.interval, use_cache=args.use_cache)
    if raw.empty:
        raise SystemExit("No OHLC rows for Backtrader.")
    pipe = FeaturePipeline(cfg)
    feats = pipe.transform(raw)

    probs = predict_probabilities(artifact, feats, regimes=feats.get("vol_regime"))
    ohlcv = raw[["open", "high", "low", "close", "volume"]].reindex(feats.index)
    back = ohlcv.join(feats[["ema_fast", "ema_slow", "atr"]], how="inner").join(probs.rename("ml_prob"), how="inner").dropna()

    report_path = resolve_path(cfg["paths"]["reports_dir"]) / "backtrader_last_run.json"

    bt_engine = GoldBacktraderEngine(cfg)
    bt_cfg = dict(cfg.get("backtrader") or {})
    strat_kw = dict(
        prob_long=args.prob_long,
        confidence_edge=float(bt_cfg.get("confidence_edge", 0.02)),
        warmup=max(50, int(cfg["vectorbt"].get("warmup_bars", 80))),
    )
    result = bt_engine.run(
        back[["open", "high", "low", "close", "volume", "ema_fast", "ema_slow", "atr", "ml_prob"]],
        report_path=str(report_path),
        strategy_kwargs=strat_kw,
    )

    print("[bt] Completed — final capital:", result["final_value"])


def cmd_dashboard(args: argparse.Namespace) -> None:
    cfg = load_app_config(args.config)
    ensure_dirs(cfg)
    dashboard_path = project_root() / "dashboard/app.py"
    print(
        "[dashboard] Run Streamlit manually:\n"
        f"  streamlit run {dashboard_path} -- --config {resolve_path(Path(args.config or 'config/default.yaml'))}",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gold AI Trading orchestration")
    p.add_argument("--config", default=str(project_root() / "config/default.yaml"))

    subs = p.add_subparsers(dest="command")

    rp = subs.add_parser("research", help="VectorBT sweep + CSV/HTML snapshot")
    rp.add_argument("--interval", default="1h")
    rp.add_argument("--symbol", default=None)
    rp.add_argument("--no-cache", action="store_false", dest="use_cache", help="force fresh download")
    rp.set_defaults(func=cmd_research, use_cache=True)

    mp = subs.add_parser("train_ml", help="Train/stack ML models + diagnostics")
    mp.add_argument("--interval", default="1h")
    mp.add_argument("--symbol", default=None)
    mp.add_argument("--no-tune", action="store_true")
    mp.set_defaults(func=cmd_train_ml, use_cache=True)

    bp = subs.add_parser("validate_bt", help="Backtest with saved ML probs")
    bp.add_argument("--interval", default="1h")
    bp.add_argument("--symbol", default=None)
    bp.add_argument("--model", default=None, help=".pkl/joblib ensemble path")
    bp.add_argument("--prob-long", type=float, dest="prob_long", default=0.52)
    bp.set_defaults(func=cmd_validate_bt, use_cache=True)

    dp = subs.add_parser("dashboard", help="Print Streamlit launch hint")
    dp.set_defaults(func=cmd_dashboard)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "func", None) is None:
        parser.print_help()
        raise SystemExit(1)
    args.func(args)


if __name__ == "__main__":
    main()
