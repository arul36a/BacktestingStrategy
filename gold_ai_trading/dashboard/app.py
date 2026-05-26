#!/usr/bin/env python3
"""
Streamlit dashboard: visualize research CSVs emitted by CLI pipelines.

Launch from repo root:
  streamlit run dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    reports_dir = repo_root / "reports"

    st.set_page_config(page_title="Gold AI Research", layout="wide")
    st.title("Gold (XAU) research dashboard")
    st.caption(
        "Loads artifacts produced by `python main.py research`, `train_ml`, "
        "`validate_bt`. For analysis only — wire live/paper feeds separately."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        sweep_path = st.text_input(
            "VectorBT sweep CSV",
            value=str(reports_dir / "research/vectorbt_param_sweep.csv"),
        )
        path = Path(sweep_path)
        if path.exists():
            df = pd.read_csv(path)
            selectable = [c for c in df.columns if c not in ("fast", "slow")]
            metric = st.selectbox("Heatmap metric", selectable or df.columns[:1])
            if {"fast", "slow"}.issubset(df.columns):
                piv = df.pivot_table(index="slow", columns="fast", values=metric)
                fig = go.Figure(
                    data=go.Heatmap(z=piv.values, x=list(map(str, piv.columns)), y=list(map(str, piv.index))),
                )
                fig.update_layout(title=f"Parameter sweep ({metric})", height=460)
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df.head(200), height=260)
        else:
            st.info("Sweep CSV missing — run `python main.py research` first.")

    with col_b:
        wf_path = st.text_input("Walk-forward diagnostics", value=str(reports_dir / "ml_walk_forward.csv"))
        if Path(wf_path).exists():
            wf = pd.read_csv(wf_path)
            st.dataframe(wf, height=320)
            if "accuracy" in wf.columns:
                st.metric("Avg fold accuracy", f"{wf['accuracy'].mean():.3f}")
        else:
            st.info("`ml_walk_forward.csv` appears after training.")

        mc_path = st.text_input(
            "Monte Carlo bootstrap summary CSV",
            value=str(reports_dir / "monte_carlo_bootstrap_summary.csv"),
        )
        if Path(mc_path).exists():
            st.dataframe(pd.read_csv(mc_path), height=240)

        st.markdown("### Paper/live trading")
        st.write(
            "Placeholder: expose tick ingestion, reconciliation, broker risk limits "
            "`EXECUTION_DRIVER=paper` pattern so research never shares code with live fills."
        )


if __name__ == "__main__":
    main()
