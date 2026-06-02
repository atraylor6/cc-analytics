"""CC Data Analytics — Streamlit app.

Run with:
    streamlit run app.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import streamlit as st

from cc_data import (
    cumulative_return,
    drawdown_chart,
    load_bloomberg_models,
    load_gips,
    load_index_data,
    load_signals,
    rolling_excess_return_chart,
    summary_stats,
)

def relative_drawdown_chart(portfolio, benchmark, ax=None):
    """Drawdown of cumulative relative performance (portfolio / benchmark)."""
    import pandas as pd
    if isinstance(portfolio, pd.Series):
        portfolio = portfolio.to_frame()
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    for col in portfolio.columns:
        s = portfolio[col].dropna()
        common = s.index.intersection(benchmark.dropna().index)
        if len(common) < 2:
            continue
        cum_p = cumulative_return(s[common])
        cum_b = cumulative_return(benchmark[common])
        ratio = cum_p / cum_b
        peak  = ratio.cummax()
        dd    = (ratio / peak - 1) * 100
        ax.fill_between(dd.index, dd.values, 0, alpha=0.4, label=str(col))
        ax.plot(dd.index, dd.values, linewidth=0.8)
    ax.set_title("Relative Drawdown vs Benchmark")
    ax.set_ylabel("Relative Drawdown (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.axhline(0, color="black", linewidth=0.7)
    ax.grid(True, alpha=0.3)
    if len(portfolio.columns) > 1:
        ax.legend(fontsize=8)
    return ax

st.set_page_config(page_title="CC Data Analytics", layout="wide", page_icon="📊")

# ── Data (cached) ─────────────────────────────────────────────────────────────

@st.cache_data
def get_all_data():
    idx       = load_index_data()
    sig       = load_signals()
    gips_ret, gips_bench = load_gips()
    bbg       = load_bloomberg_models()
    return idx, sig, gips_ret, gips_bench, bbg

idx, sig, gips_ret, gips_bench, bbg = get_all_data()

idx_flat = idx.copy()
idx_flat.columns = idx_flat.columns.get_level_values("name")

# Blended ACWI/AGG benchmarks
_acwi = sig["ACWI"]
_agg  = sig["AGG"]
BLENDED_BENCHMARKS = {
    "100/0 ACWI/AGG":  1.00 * _acwi + 0.00 * _agg,
    "80/20 ACWI/AGG":  0.80 * _acwi + 0.20 * _agg,
    "60/40 ACWI/AGG":  0.60 * _acwi + 0.40 * _agg,
    "40/60 ACWI/AGG":  0.40 * _acwi + 0.60 * _agg,
}

SIGNAL_MODEL_COLS = [
    "Tier I Model", "GV Model", "EAFE Model", "EM Model",
    "T1F Model", "OMFL Model", "Tier II", "Market Risk",
]
SIGNAL_BENCH_COLS = [
    "AGG", "ACWI", "EAFE", "EM", "Gold",
    "Growth", "Value", "Momentum", "Min Vol", "R1",
]

SOURCE_DATA = {
    "Index Returns":           idx_flat,
    "Tier II Model Returns":   sig[SIGNAL_MODEL_COLS],
    "GIPS Returns":            gips_ret,
    "Bloomberg Model Returns": bbg,
}

ABS_STATS = [
    "Ann. Return (%)", "Ann. Std Dev (%)",
    "Sharpe", "Max Drawdown (%)", "Win Rate (%)", "Months",
]
REL_STATS = [
    "Excess Return (%)", "Tracking Error (%)",
    "Downside Dev (%)", "Sortino", "Info Ratio",
]
STAT_FMT = {
    "Ann. Return (%)":    "{:.2f}",
    "Ann. Std Dev (%)":   "{:.2f}",
    "Downside Dev (%)":   "{:.2f}",
    "Sharpe":             "{:.3f}",
    "Sortino":            "{:.3f}",
    "Max Drawdown (%)":   "{:.2f}",
    "Win Rate (%)":       "{:.1f}",
    "Months":             "{:.0f}",
    "Excess Return (%)":  "{:.2f}",
    "Tracking Error (%)": "{:.2f}",
    "Info Ratio":         "{:.3f}",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 CC Data Analytics")
    st.divider()

    # 1 — Data source
    data_source = st.radio("**Data Source**", list(SOURCE_DATA.keys()))
    data = SOURCE_DATA[data_source]
    st.divider()

    # 2 — Series
    all_series = data.columns.tolist()
    selected_series = st.multiselect(
        "**Series**", all_series,
        default=all_series[:min(3, len(all_series))],
    )
    st.divider()

    # 3 — Date range
    d_min = data.index.min().date()
    d_max = data.index.max().date()
    date_range = st.date_input(
        "**Date Range**",
        value=(d_min, d_max),
        min_value=d_min,
        max_value=d_max,
    )
    st.divider()

    # 4 — Benchmark
    bench_map: dict[str, tuple[str, str]] = {}
    if data_source == "GIPS Returns":
        for c in gips_bench.columns:
            bench_map[f"GIPS | {c}"] = ("gips", c)
    for n in BLENDED_BENCHMARKS:
        bench_map[f"Blend | {n}"] = ("blend", n)
    for n in SIGNAL_BENCH_COLS:
        bench_map[f"Signals | {n}"] = ("signals", n)
    for n in idx_flat.columns:
        bench_map[f"Index | {n}"] = ("index", n)

    benchmark_key = st.selectbox("**Benchmark**", ["None"] + list(bench_map.keys()))
    st.divider()

    # 5 — Statistics
    st.markdown("**Statistics**")
    selected_abs = st.multiselect(
        "Absolute", ABS_STATS,
        default=["Ann. Return (%)", "Ann. Std Dev (%)", "Sharpe", "Max Drawdown (%)"],
    )
    selected_rel: list[str] = []
    if benchmark_key != "None":
        selected_rel = st.multiselect("Relative (vs benchmark)", REL_STATS, default=REL_STATS)
    st.divider()

    # 6 — Charts
    st.markdown("**Charts**")

    show_drawdown     = st.checkbox("Absolute Drawdown")
    show_rel_drawdown = st.checkbox("Relative Drawdown vs Benchmark")
    if show_rel_drawdown and benchmark_key == "None":
        st.caption("  ⚠ Select a benchmark above.")

    show_ratio = st.checkbox("Ratio Chart")
    if show_ratio and benchmark_key == "None":
        st.caption("  ⚠ Select a benchmark above.")

    show_rolling = st.checkbox("Rolling Excess Return")
    roll_windows: list[int] = [12, 36]
    if show_rolling:
        if benchmark_key == "None":
            st.caption("  ⚠ Select a benchmark above.")
        else:
            roll_windows = st.multiselect(
                "  Windows (months)", [12, 24, 36, 60], default=[12, 36]
            ) or [12, 36]

    st.divider()
    run = st.button("▶  Run Analysis", type="primary", use_container_width=True)

# ── Guard: nothing to show yet ────────────────────────────────────────────────

if not run:
    st.markdown("### Configure your analysis in the sidebar, then click **Run Analysis**.")
    st.stop()

if not selected_series:
    st.warning("Select at least one series in the sidebar.")
    st.stop()

if not isinstance(date_range, (list, tuple)) or len(date_range) != 2:
    st.warning("Select a complete date range (start and end).")
    st.stop()

start_date, end_date = date_range[0], date_range[1]

# ── Slice data ────────────────────────────────────────────────────────────────

returns = data[selected_series].loc[start_date:end_date].dropna(how="all")

if returns.empty:
    st.error("No data in the selected date range.")
    st.stop()

# ── Resolve benchmark ─────────────────────────────────────────────────────────

benchmark: pd.Series | None = None
if benchmark_key != "None":
    src, name = bench_map[benchmark_key]
    raw_bench = (
        gips_bench[name]          if src == "gips" else
        BLENDED_BENCHMARKS[name]  if src == "blend" else
        sig[name]                 if src == "signals" else
        idx_flat[name]
    )
    benchmark = raw_bench.loc[start_date:end_date].dropna()

# ── Statistics table ──────────────────────────────────────────────────────────

show_stats = selected_abs or selected_rel
if show_stats:
    stats_df = summary_stats(returns, benchmark=benchmark)
    cols = [c for c in selected_abs + selected_rel if c in stats_df.columns]
    if cols:
        st.subheader("Statistics")
        fmt = {k: v for k, v in STAT_FMT.items() if k in cols}
        st.dataframe(
            stats_df[cols].style.format(fmt),
            use_container_width=True,
        )

# ── Absolute drawdown chart ───────────────────────────────────────────────────

if show_drawdown:
    st.subheader("Absolute Drawdown")
    fig, ax = plt.subplots(figsize=(12, 4))
    drawdown_chart(returns, ax=ax)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── Relative drawdown chart ───────────────────────────────────────────────────

if show_rel_drawdown:
    if benchmark is None:
        st.warning("Select a benchmark for relative drawdown.")
    else:
        bench_name = bench_map[benchmark_key][1]
        st.subheader(f"Relative Drawdown vs {bench_name}")
        fig, ax = plt.subplots(figsize=(12, 4))
        relative_drawdown_chart(returns, benchmark, ax=ax)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ── Ratio chart ───────────────────────────────────────────────────────────────

if show_ratio:
    if benchmark is None:
        st.warning("Select a benchmark to generate a ratio chart.")
    else:
        bench_name = bench_map[benchmark_key][1]
        st.subheader(f"Ratio vs {bench_name}")
        fig, ax = plt.subplots(figsize=(12, 4))
        for series in selected_series:
            s = returns[series].dropna()
            common = s.index.intersection(benchmark.index)
            if len(common) < 2:
                continue
            cum_s = cumulative_return(s[common])
            cum_b = cumulative_return(benchmark[common])
            ratio = cum_s / cum_b
            ax.plot(ratio.index, ratio.values, label=series, linewidth=1.5)
        ax.axhline(1.0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.set_ylabel(f"Ratio vs {bench_name} (rebased to 1.0)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ── Rolling excess return chart ───────────────────────────────────────────────

if show_rolling and benchmark is not None:
    st.subheader("Rolling Excess Return")
    for series in selected_series:
        s = returns[series].dropna()
        if s.empty:
            continue
        fig, ax = plt.subplots(figsize=(12, 4))
        rolling_excess_return_chart(s, benchmark, windows=roll_windows, label=series, ax=ax)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
