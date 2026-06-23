"""CC Data Analytics — Streamlit app.

Run with:
    streamlit run app.py
"""

import calendar
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date

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
import plotly.graph_objects as go
from attribution import (
    NEUTRAL_EQUITY,
    EQUITY_SUBMODELS,
    GEO_SUBMODELS,
    R1000_SUBMODELS,
    compute_attribution_table,
    compute_monthly_attribution,
    compute_position_attribution,
    compute_tier1_trades,
    compute_aftertax_summary,
    compute_positioning,
)


def relative_drawdown_chart(portfolio, benchmark, ax=None):
    """Drawdown of cumulative relative performance (portfolio / benchmark)."""
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


# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data
def get_all_data():
    idx           = load_index_data()
    sig           = load_signals()
    gips_ret, gips_bench = load_gips()
    bbg           = load_bloomberg_models()
    return idx, sig, gips_ret, gips_bench, bbg

@st.cache_data
def get_monthly_attribution(neutral_equity: float, rebal_sheet: str, equity_only: bool):
    return compute_monthly_attribution(neutral_equity=neutral_equity,
                                       rebal_sheet=rebal_sheet, equity_only=equity_only)

@st.cache_data
def get_period_attribution(start: str, end: str, neutral_equity: float,
                           rebal_sheet: str, equity_only: bool):
    return compute_attribution_table(start, end, neutral_equity=neutral_equity,
                                     rebal_sheet=rebal_sheet, equity_only=equity_only)

@st.cache_data
def get_position_attribution(start: str, end: str, neutral_equity: float,
                              rebal_sheet: str, equity_only: bool):
    return compute_position_attribution(start, end, neutral_equity=neutral_equity,
                                        rebal_sheet=rebal_sheet, equity_only=equity_only)

@st.cache_data
def get_tier1_trades(neutral_equity: float, rebal_sheet: str):
    return compute_tier1_trades(neutral_equity=neutral_equity, rebal_sheet=rebal_sheet)

@st.cache_data
def get_aftertax_summary(start: str, end: str, neutral_equity: float,
                          ltcg_rate: float, stcg_rate: float, accrual: bool = False):
    return compute_aftertax_summary(start, end, neutral_equity=neutral_equity,
                                    ltcg_rate=ltcg_rate, stcg_rate=stcg_rate,
                                    accrual=accrual)

def _tbl_h(df):
    return 36 * (len(df) + 1) + 3


idx, sig, gips_ret, gips_bench, bbg = get_all_data()
idx_flat = idx.copy()
idx_flat.columns = idx_flat.columns.get_level_values("name")

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

PORTFOLIO_OPTIONS = {
    "Aggressive (100/0)":   1.00,
    "Growth (80/20)":       0.80,
    "Balanced (60/40)":     0.60,
    "Conservative (40/60)": 0.40,
}

MODEL_CONFIGS = {
    "PM Balanced SGA":           {"rebal_sheet": "Rebalances",    "equity_only": False},
    "PM Balanced":               {"rebal_sheet": "TE Rebalances", "equity_only": False},
    "Market Risk":               {"rebal_sheet": "Rebalances",    "equity_only": True},
    "Market Risk Unconstrained": {"rebal_sheet": "TE Rebalances", "equity_only": True},
}

# Display order for Level 2 sub-model rows
SUBMODEL_ORDER = ["EAFE", "EM", "GV", "OMFL", "SGA", "T1F",
                  "Sector", "SmallCap", "Art", "REIT", "Gold"]


# ── Sidebar — page navigation ─────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 CC Data Analytics")
    st.divider()
    page = st.radio("**Page**", ["Analytics", "Attribution"])
    st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

if page == "Analytics":

    with st.sidebar:
        data_source = st.radio("**Data Source**", list(SOURCE_DATA.keys()))
        data = SOURCE_DATA[data_source]
        st.divider()

        all_series = data.columns.tolist()
        selected_series = st.multiselect(
            "**Series**", all_series,
            default=all_series[:min(3, len(all_series))],
        )
        st.divider()

        d_min = data.index.min().date()
        d_max = data.index.max().date()
        date_range = st.date_input(
            "**Date Range**",
            value=(d_min, d_max),
            min_value=d_min,
            max_value=d_max,
        )
        st.divider()

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

        st.markdown("**Statistics**")
        selected_abs = st.multiselect(
            "Absolute", ABS_STATS,
            default=["Ann. Return (%)", "Ann. Std Dev (%)", "Sharpe", "Max Drawdown (%)"],
        )
        selected_rel: list[str] = []
        if benchmark_key != "None":
            selected_rel = st.multiselect("Relative (vs benchmark)", REL_STATS, default=REL_STATS)
        st.divider()

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
    returns = data[selected_series].loc[start_date:end_date].dropna(how="all")

    if returns.empty:
        st.error("No data in the selected date range.")
        st.stop()

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

    if show_drawdown:
        st.subheader("Absolute Drawdown")
        fig, ax = plt.subplots(figsize=(12, 4))
        drawdown_chart(returns, ax=ax)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

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


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ATTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Attribution":

    neutral_eq = 0.60   # Balanced (60/40)

    _today = date.today()

    def _safe_date(year, month, day):
        return date(year, month, min(day, calendar.monthrange(year, month)[1]))

    if "attr_start_d" not in st.session_state:
        st.session_state["attr_start_d"] = _safe_date(_today.year - 1, _today.month, _today.day)
    if "attr_end_d" not in st.session_state:
        st.session_state["attr_end_d"] = _today

    with st.sidebar:
        model_name = st.radio("**Model**", list(MODEL_CONFIGS.keys()))
        _mcfg      = MODEL_CONFIGS[model_name]
        rebal_sheet = _mcfg["rebal_sheet"]
        equity_only = _mcfg["equity_only"]
        _bond_label = 'AGG' if rebal_sheet == 'TE Rebalances' else 'MUB'
        st.divider()

        _available_views = ["Period Summary", "Monthly Breakdown"]
        if model_name == "PM Balanced SGA":
            _available_views += ["After-Tax", "Positioning"]
        view = st.radio("**View**", _available_views)
        st.divider()

        st.markdown("**Date Range**")
        bcols = st.columns(4)
        if bcols[0].button("YTD",  use_container_width=True):
            st.session_state["attr_start_d"] = date(_today.year, 1, 1)
            st.session_state["attr_end_d"]   = _today
        if bcols[1].button("1Y",   use_container_width=True):
            st.session_state["attr_start_d"] = _safe_date(_today.year - 1, _today.month, _today.day)
            st.session_state["attr_end_d"]   = _today
        if bcols[2].button("3Y",   use_container_width=True):
            st.session_state["attr_start_d"] = _safe_date(_today.year - 3, _today.month, _today.day)
            st.session_state["attr_end_d"]   = _today
        if bcols[3].button("5Y",   use_container_width=True):
            st.session_state["attr_start_d"] = _safe_date(_today.year - 5, _today.month, _today.day)
            st.session_state["attr_end_d"]   = _today

        attr_start = st.date_input("Start", key="attr_start_d")
        attr_end   = st.date_input("End",   key="attr_end_d")
        st.divider()

        annualize = st.checkbox("Annualize")
        st.divider()
        if view not in ("After-Tax", "Positioning"):
            run_attr = st.button("▶  Run Attribution", type="primary", use_container_width=True)
        else:
            run_attr = False

    _bm_label = "ACWI" if equity_only else "Balanced (60/40)"
    st.header(f"Return Attribution — {model_name}  |  Benchmark: {_bm_label}")

    if not run_attr and view not in ("After-Tax", "Positioning"):
        st.markdown("Select a date range in the sidebar, then click **Run Attribution**.")
        st.stop()

    # ── Shared date / annualisation helpers ───────────────────────────────────

    period_days = (attr_end - attr_start).days
    years       = max(period_days, 1) / 365.25

    def _geo_ann(r):
        return ((1 + r / 100) ** (1 / years) - 1) * 100

    def _ann_ret(r):
        return _geo_ann(r) if annualize else r

    # ── Period Summary ─────────────────────────────────────────────────────────

    if view == "Period Summary":

        with st.spinner("Computing attribution..."):
            try:
                tbl = get_period_attribution(str(attr_start), str(attr_end), neutral_eq,
                                             rebal_sheet, equity_only)
                pos = get_position_attribution(str(attr_start), str(attr_end), neutral_eq,
                                               rebal_sheet, equity_only)
            except ValueError as e:
                st.error(str(e))
                st.stop()

        if annualize and period_days < 7:
            st.warning("Annualization is not meaningful for periods under 7 days.")
            st.stop()

        ret_label = "Ann. Return" if annualize else "Return"

        # ── Summary metrics ────────────────────────────────────────────────────

        port_val   = float(tbl.loc["Portfolio Return", "Total"]) if "Portfolio Return" in tbl.index else float("nan")
        bench_val  = float(tbl.loc["Benchmark Return", "Total"]) if "Benchmark Return" in tbl.index else float("nan")
        excess_val = float(tbl.loc["Excess Return",    "Total"]) if "Excess Return"    in tbl.index else float("nan")

        c1, c2, c3 = st.columns(3)
        c1.metric(f"Portfolio {ret_label}", f"{_ann_ret(port_val):.2f}%")
        c2.metric(f"Benchmark {ret_label}", f"{_ann_ret(bench_val):.2f}%")
        ann_exc = _ann_ret(excess_val)
        c3.metric("Excess Return", f"{ann_exc:.2f}%", delta=f"{ann_exc:.2f}%",
                  delta_color="normal" if ann_exc >= 0 else "inverse")

        st.divider()

        # ── Export ─────────────────────────────────────────────────────────────

        def _build_excel(tbl_, pos_):
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                tbl_.to_excel(writer, sheet_name='Summary')
                if 'Tier1' in pos_:
                    pos_['Tier1'].to_excel(writer, sheet_name='Tier I')
                if 'FixedIncome' in pos_ and not pos_['FixedIncome'].empty:
                    pos_['FixedIncome'].to_excel(writer, sheet_name='Fixed Income')
                for sm in SUBMODEL_ORDER:
                    if sm in pos_ and not pos_[sm].empty:
                        pos_[sm].to_excel(writer, sheet_name=sm[:31])
            buf.seek(0)
            return buf.getvalue()

        st.download_button(
            label="Download Excel",
            data=_build_excel(tbl, pos),
            file_name=f"attribution_{attr_start}_{attr_end}_Balanced.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        ATTR_COLS = ["Allocation (Passive)", "Allocation (Active)", "Selection", "Total"]
        STAT_COLS = ["Wt%", "Bm Wt%", "Ret%", "Bm Ret%", "Contrib%", "Bm Contrib%"]
        fmt_attr  = {
            "Wt%": "{:.2f}", "Bm Wt%": "{:.2f}",
            "Ret%": "{:.2f}", "Bm Ret%": "{:.2f}",
            "Contrib%": "{:.3f}", "Bm Contrib%": "{:.3f}",
            **{c: "{:.3f}" for c in ATTR_COLS},
        }
        fmt_pos   = {"Avg Wt%": "{:.1f}", "Return%": "{:.2f}",
                     "Active Return%": "{:.2f}", "Contribution%": "{:.3f}"}

        def _pos_style(df):
            return df.style.format(fmt_pos, na_rep="--")

        def _signed(v):
            return f"+{v:.2f}%" if v >= 0 else f"{v:.2f}%"

        active_sms = [s for s in SUBMODEL_ORDER
                      if s in tbl.index and abs(float(tbl.loc[s, "Total"])) > 0.0005]

        # ── Attribution Summary ─────────────────────────────────────────────────
        US_R1000_SMs  = [s for s in ("GV", "OMFL", "Sector") if s in active_sms]
        equity_detail = ([s for s in ("EAFE", "EM") if s in tbl.index]
                         + US_R1000_SMs
                         + [s for s in SUBMODEL_ORDER
                            if s not in ("EAFE", "EM", "GV", "OMFL", "Sector") and s in active_sms])
        _known_rows    = ({"Portfolio Return", "Benchmark Return", "Excess Return",
                           "Tier I (equity/bond)", "Fixed Income", "Total Stock", "Total Bond"}
                          | EQUITY_SUBMODELS)
        bond_detail    = [r for r in tbl.index if r not in _known_rows]

        MAIN_ROWS  = [r for r in ("Tier I (equity/bond)", "Total Stock", "Total Bond")
                      if r in tbl.index]
        DISP_NAMES = {"Tier I (equity/bond)": "Tier I — Equity/Bond Timing"}

        MAIN_STAT  = ["Wt%", "Bm Wt%", "Ret%", "Bm Ret%", "Contrib%", "Bm Contrib%"]
        MAIN_ATTR  = ["Allocation (Passive)", "Allocation (Active)", "Selection", "Total"]
        main_cols  = [c for c in MAIN_STAT + MAIN_ATTR if c in tbl.columns]
        DETL_COLS  = [c for c in ["Wt%", "Ret%", "Contrib%"] + MAIN_ATTR if c in tbl.columns]

        fmt_main = {
            "Wt%": "{:.2f}", "Bm Wt%": "{:.2f}",
            "Ret%": "{:.2f}", "Bm Ret%": "{:.2f}",
            "Contrib%": "{:.3f}", "Bm Contrib%": "{:.3f}",
            **{c: "{:.3f}" for c in MAIN_ATTR},
        }
        fmt_detl = {"Wt%": "{:.2f}", "Ret%": "{:.2f}", "Contrib%": "{:.3f}",
                    **{c: "{:.3f}" for c in MAIN_ATTR}}

        main_df = tbl.loc[MAIN_ROWS, main_cols].copy()
        sum_c   = [c for c in ["Contrib%", "Bm Contrib%"] + MAIN_ATTR if c in main_df.columns]
        total_r = main_df[sum_c].fillna(0).sum().rename("Total")
        main_df = pd.concat([main_df, total_r.to_frame().T])
        main_df.index = [DISP_NAMES.get(r, r) for r in main_df.index]
        main_df.index.name = "Effect"

        st.subheader("Attribution Summary")
        st.dataframe(main_df.style.format(fmt_main, na_rep="--"),
                     use_container_width=True, height=_tbl_h(main_df))

        if equity_detail:
            with st.expander("Stock Components"):
                eq_df = tbl.loc[[r for r in equity_detail if r in tbl.index], DETL_COLS].copy()
                st.dataframe(eq_df.style.format(fmt_detl, na_rep="--"),
                             use_container_width=True, height=_tbl_h(eq_df))

        if bond_detail:
            with st.expander("Bond Components"):
                BOND_COLS = [c for c in ["Wt%", "Ret%", "Bm Ret%", "Contrib%", "Selection"]
                             if c in tbl.columns]
                fmt_bond  = {"Wt%": "{:.2f}", "Ret%": "{:.2f}", "Bm Ret%": "{:.2f}",
                             "Contrib%": "{:.3f}", "Selection": "{:.3f}"}
                bd_df = tbl.loc[[r for r in bond_detail if r in tbl.index], BOND_COLS].copy()
                st.dataframe(bd_df.style.format(fmt_bond, na_rep="--"),
                             use_container_width=True, height=_tbl_h(bd_df))

        st.caption(
            "Wt% / Bm Wt%: avg portfolio and benchmark weights.  "
            "Ret% / Bm Ret%: compounded period return.  "
            "Contrib% / Bm Contrib%: Carino-linked contribution to total return.  "
            "Attribution effects: Carino geometric excess return."
        )

        st.divider()

        # ── Position drill-down expanders (flat) ────────────────────────────────

        st.subheader("Position Detail")

        if not equity_only:
            with st.expander("Tier I — Equity/Bond Timing"):
                if "Tier1" in pos:
                    fmt_t1 = {"Avg Wt%": "{:.1f}", "Neutral Wt%": "{:.1f}",
                              "Active OW%": "{:.1f}", "Period Ret%": "{:.2f}"}
                    st.dataframe(pos["Tier1"].style.format(fmt_t1),
                                 use_container_width=True, height=_tbl_h(pos["Tier1"]))
                st.caption(f"Effect = avg daily active equity OW × (ACWI − {_bond_label}) spread, Carino-linked")

            with st.expander("Fixed Income — Selection"):
                fi_df = pos.get("FixedIncome", pd.DataFrame())
                if not fi_df.empty:
                    st.dataframe(_pos_style(fi_df), use_container_width=True, height=_tbl_h(fi_df))
                else:
                    st.info("No fixed income positions in this period.")
                st.caption(f"Active Return% vs {_bond_label}")

        for sm in ([s for s in ("EAFE", "EM") if s in tbl.index]
                   + US_R1000_SMs
                   + [s for s in SUBMODEL_ORDER
                      if s not in ("EAFE", "EM", "GV", "OMFL", "Sector") and s in active_sms]):
            sm_val = float(tbl.loc[sm, "Total"])
            with st.expander(f"{sm}   {_signed(sm_val)}"):
                sm_pos = pos.get(sm, pd.DataFrame())
                if not sm_pos.empty:
                    st.dataframe(_pos_style(sm_pos), use_container_width=True,
                                 height=_tbl_h(sm_pos))
                else:
                    st.info("No position data.")
            bench_label = "Russell 1000 (IWB)" if sm in R1000_SUBMODELS else "ACWI"
            st.caption(f"Active Return% vs {bench_label}")

        st.divider()

        # ── Contributions bar chart ────────────────────────────────────────────

        st.subheader("Sub-model Contributions")
        chart_rows = ([s for s in ("EAFE", "EM") if s in tbl.index]
                      + US_R1000_SMs
                      + [s for s in SUBMODEL_ORDER
                         if s not in ("EAFE", "EM", "GV", "OMFL", "Sector") and s in active_sms])
        if chart_rows:
            totals = tbl.loc[chart_rows, "Total"].fillna(0)
            colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in totals]

            fig, ax = plt.subplots(figsize=(max(6, len(totals) * 1.1), 2.8))
            bars = ax.bar(totals.index, totals.values, color=colors, width=0.55, edgecolor="white")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylabel("Contribution (%)")
            ax.grid(True, axis="y", alpha=0.3)
            ax.set_axisbelow(True)
            for bar, val in zip(bars, totals.values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val,
                        f"{val:.2f}",
                        ha="center",
                        va="bottom" if val >= 0 else "top",
                        fontsize=8)
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin * 1.15, ymax * 1.15)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # ── Monthly Breakdown ──────────────────────────────────────────────────────

    elif view == "Monthly Breakdown":

        with st.spinner("Computing monthly attribution..."):
            monthly = get_monthly_attribution(neutral_eq, rebal_sheet, equity_only)

        monthly_filtered = monthly[(monthly.index > pd.Timestamp(attr_start)) &
                                    (monthly.index <= pd.Timestamp(attr_end))]

        if monthly_filtered.empty:
            st.warning("No monthly data in the selected date range.")
            st.stop()

        # Primary columns to display
        # Only include sub-models that have at least one non-zero month
        active_sms = [m for m in SUBMODEL_ORDER
                      if m in monthly_filtered.columns
                      and monthly_filtered[m].abs().max() > 0.001]
        _l1_cols = [] if equity_only else ["tier1", "fi_selection"]
        primary_cols = (
            ["portfolio_return", "benchmark_return", "excess_return"]
            + _l1_cols
            + active_sms
        )
        display_monthly = monthly_filtered[primary_cols].copy()
        display_monthly.index = display_monthly.index.strftime("%b %Y")
        display_monthly.index.name = "Month"

        # Rename base columns; sub-model names (EAFE, EM, GV, etc.) keep their original casing
        col_labels = {
            "portfolio_return": "Portfolio",
            "benchmark_return": "Benchmark",
            "excess_return":    "Excess",
            "tier1":            "Tier I",
            "fi_selection":     "FI Selection",
        }
        display_monthly.columns = [col_labels.get(c, c) for c in display_monthly.columns]

        st.subheader("Monthly Attribution Effects (%)")
        st.dataframe(
            display_monthly.style
            .format("{:.2f}", na_rep="--"),
            use_container_width=True,
            height=600,
        )

        # ── Cumulative excess return chart ─────────────────────────────────────

        st.divider()
        st.subheader("Cumulative Excess Return")

        excess_series = monthly_filtered["excess_return"].dropna()
        if not excess_series.empty:
            cumulative = excess_series.cumsum()
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.bar(cumulative.index, excess_series.values,
                   color=["#2ecc71" if v >= 0 else "#e74c3c" for v in excess_series.values],
                   width=20, alpha=0.6, label="Monthly excess")
            ax.plot(cumulative.index, cumulative.values, color="navy",
                    linewidth=1.5, label="Cumulative excess")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylabel("Excess Return (%)")
            ax.legend(fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # ── Secondary: EAFE/EM alloc vs sel ────────────────────────────────────

        geo_cols = [c for c in ["EAFE_alloc", "EAFE_sel", "EM_alloc", "EM_sel"]
                    if c in monthly_filtered.columns]
        if geo_cols:
            st.divider()
            st.subheader("EAFE / EM Active Allocation vs Selection")
            geo_display = monthly_filtered[geo_cols].copy()
            geo_display.index = geo_display.index.strftime("%b %Y")
            geo_display.columns = [c.replace("_alloc", " Active Alloc")
                                     .replace("_sel",  " Selection")
                                    for c in geo_cols]
            st.dataframe(
                geo_display.style
                .format("{:.3f}", na_rep="--")
                .background_gradient(cmap="RdYlGn", axis=None, vmin=-0.5, vmax=0.5),
                use_container_width=True,
                height=400,
            )

        # ── Tier I trade log (standard mode only) ───────────────────────────────

        if equity_only:
            st.stop()

        st.divider()
        st.subheader("Tier I — Equity/Bond Timing Trades")

        trades_all = get_tier1_trades(neutral_eq, rebal_sheet)
        trades = trades_all.loc[str(attr_start):str(attr_end)].copy()

        if not trades.empty:
            n_active   = (trades["Result"] != "—").sum()
            n_worked   = (trades["Result"] == "✓").sum()
            n_didnt    = (trades["Result"] == "✗").sum()

            m1, m2, m3 = st.columns(3)
            m1.metric("Months with Active Bet", int(n_active))
            m2.metric("Worked",     int(n_worked))
            m3.metric("Didn't Work", int(n_didnt))

            trades.index = trades.index.strftime("%b %Y")
            trades.index.name = "Month"

            def _style_trades(df):
                styles = pd.DataFrame("", index=df.index, columns=df.columns)
                for idx in df.index:
                    r = df.loc[idx, "Result"]
                    if r == "✓":
                        styles.loc[idx, "Result"] = "background-color: #d4edda; color: #155724"
                    elif r == "✗":
                        styles.loc[idx, "Result"] = "background-color: #f8d7da; color: #721c24"
                    tier1 = df.loc[idx, "Tier I%"]
                    if isinstance(tier1, float):
                        styles.loc[idx, "Tier I%"] = (
                            "color: #155724" if tier1 > 0 else
                            "color: #721c24" if tier1 < 0 else ""
                        )
                return styles

            st.dataframe(
                trades.style
                .format({"Active Eq OW%": "{:.1f}", "ACWI%": "{:.2f}",
                         f"{_bond_label}%": "{:.2f}", "Spread%": "{:.2f}", "Tier I%": "{:.3f}"})
                .apply(_style_trades, axis=None),
                use_container_width=True,
                height=min(36 * (len(trades) + 1) + 3, 600),
            )
            st.caption(
                "Active Eq OW% = portfolio equity wt − 60% neutral  |  "
                f"Spread = ACWI − {_bond_label} for the month  |  "
                "✓ = bet and outcome aligned  |  ✗ = bet and outcome misaligned  |  — = near-neutral position"
            )

    # ── After-Tax ──────────────────────────────────────────────────────────────

    elif view == "After-Tax":

        st.subheader("After-Tax Return Analysis")
        st.caption(
            "Realised capital gains from rebalancing trades only — dividends excluded.  "
            "FIFO lot matching.  Losses per lot are not offset against gains (conservative).  "
            "Benchmark treated as buy-and-hold (no realised gains)."
        )

        with st.sidebar:
            st.markdown("**Tax Rates**")
            ltcg = st.number_input("Long-term CGT rate (%)",  value=23.8, step=0.1,
                                   min_value=0.0, max_value=60.0) / 100
            stcg = st.number_input("Short-term CGT rate (%)", value=43.8, step=0.1,
                                   min_value=0.0, max_value=60.0) / 100
            st.divider()
            st.markdown("**Method**")
            tax_method = st.radio(
                "Tax recognition",
                ["Realization", "Accrual (mark-to-market)"],
                help=(
                    "**Realization**: taxes deducted when a gain is actually sold.  \n"
                    "A 1-year window that sells a 10-year holding bears the full "
                    "embedded gain as a cost in that year.  \n\n"
                    "**Accrual**: deferred-tax liability (unrealised gains × rate) "
                    "is subtracted from portfolio value every day, so tax drag "
                    "accrues smoothly as gains build — regardless of when sold."
                ),
                label_visibility="collapsed",
            )
            accrual = tax_method == "Accrual (mark-to-market)"

        run_at = st.button("▶  Run After-Tax", type="primary", use_container_width=True)

        if run_at:
            spinner_msg = ("Replaying full portfolio history for cost-basis tracking "
                           + ("+ daily DTL…" if accrual else "…"))
            with st.spinner(spinner_msg):
                at = get_aftertax_summary(
                    str(attr_start), str(attr_end),
                    neutral_eq, ltcg, stcg, accrual,
                )

            def _disp_e(e):
                if not annualize:
                    return e
                return _geo_ann(at['aftertax_return']) - _geo_ann(at['pretax_return'])

            ret_sfx = "/yr" if annualize else ""

            # ── Summary metrics ──
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Pre-Tax Return",
                      f"{_ann_ret(at['pretax_return']):+.2f}%{ret_sfx}",
                      f"{_geo_ann(at['pretax_return']):+.2f}%/yr" if not annualize else None)
            c2.metric("After-Tax Return",
                      f"{_ann_ret(at['aftertax_return']):+.2f}%{ret_sfx}",
                      f"{_geo_ann(at['aftertax_return']):+.2f}%/yr" if not annualize else None)
            c3.metric("Benchmark Return",
                      f"{_ann_ret(at['benchmark_return']):+.2f}%{ret_sfx}",
                      f"{_geo_ann(at['benchmark_return']):+.2f}%/yr" if not annualize else None)
            c4.metric("Tax Drag",
                      f"{_disp_e(at['tax_drag']):+.2f}%{ret_sfx}",
                      delta_color="inverse")

            st.divider()

            # ── Tax breakdown ──
            st.subheader("Tax Paid Breakdown")
            tax_df = pd.DataFrame({
                "Category":         ["Long-term gains (≥365 days)", "Short-term gains (<365 days)", "Total"],
                "Tax (% of portf)": [at['lt_tax_pct'], at['st_tax_pct'], at['total_tax_pct']],
                "Rate applied":     [f"{ltcg*100:.1f}%", f"{stcg*100:.1f}%", "—"],
            }).set_index("Category")
            st.dataframe(tax_df.style.format({"Tax (% of portf)": "{:.3f}%"}),
                         use_container_width=True, height=_tbl_h(tax_df))

            st.divider()

            # ── Pre-tax vs after-tax by sub-model ──
            log = at['trade_log']

            tbl_pt = get_period_attribution(str(attr_start), str(attr_end), neutral_eq,
                                             rebal_sheet, equity_only)

            # Tax paid per sub-model, split between Tier I and within-model trades.
            # tier1_frac = fraction of each sell driven by the equity/bond allocation
            # shift (Tier I decision) vs. within-equity reweighting (sub-model decision).
            SM_LABEL = {'FixedIncome': 'Fixed Income'}
            if not log.empty:
                lg = log.copy()
                frac = lg['tier1_frac'] if 'tier1_frac' in lg.columns else 0.0
                lg['_sm_tax']    = lg['tax_%'] * (1 - frac)
                lg['_tier1_tax'] = lg['tax_%'] * frac
                tax_by_sm = (lg.groupby('sub_model')['_sm_tax'].sum()
                             .rename(index=lambda x: SM_LABEL.get(x, x)))
                tax_by_sm['Tier I (equity/bond)'] = float(lg['_tier1_tax'].sum())
            else:
                tax_by_sm = pd.Series(dtype=float)

            SKIP = {'Portfolio Return', 'Benchmark Return', 'Excess Return'}
            at_rows = []
            for effect in tbl_pt.index:
                if effect in SKIP:
                    continue
                pretax   = float(tbl_pt.loc[effect, 'Total'])
                tax_cost = float(tax_by_sm.get(effect, 0.0))
                at_rows.append({
                    'Effect':             effect,
                    'Pre-Tax Contrib%':   round(pretax,            3),
                    'Tax Cost%':          round(-tax_cost,         3),
                    'After-Tax Contrib%': round(pretax - tax_cost, 3),
                })

            at_sm_df = pd.DataFrame(at_rows).set_index('Effect')

            # Total row — use geometric excess so it reconciles with the per-effect column sum
            pretax_excess   = float(tbl_pt.loc['Excess Return', 'Total']) if 'Excess Return' in tbl_pt.index else float('nan')
            aftertax_excess = ((1 + at['aftertax_return'] / 100) / (1 + at['benchmark_return'] / 100) - 1) * 100
            total_row = pd.DataFrame([{
                'Pre-Tax Contrib%':   round(pretax_excess,          3),
                'Tax Cost%':          round(-at['total_tax_pct'],   3),
                'After-Tax Contrib%': round(aftertax_excess,        3),
            }], index=pd.Index(['Total'], name='Effect'))
            at_sm_df = pd.concat([at_sm_df, total_row])
            at_sm_df.index.name = 'Effect'

            st.subheader("Pre-Tax vs After-Tax by Sub-Model")
            sm_note = ("Tax Cost% = realized taxes from each model's trades." if not accrual
                       else "Tax Cost% = realized taxes from each model's trades "
                            "(realization basis — accrual DTL is not attributable by sub-model).")
            st.caption(
                "Pre-Tax Contrib% = Carino-linked attribution effect vs benchmark.  "
                + sm_note + "  After-Tax Contrib% = Pre-Tax − Tax Cost."
            )

            st.dataframe(
                at_sm_df.style.format('{:+.3f}%', na_rep='--'),
                use_container_width=True,
                height=_tbl_h(at_sm_df),
            )

            st.divider()

            # ── Full trade log ──
            st.subheader("Realised Gain / Tax Trade Log")
            if log.empty:
                st.info("No realised gains in the selected period.")
            else:
                log_display = log.copy()
                log_display['date']     = log_display['date'].dt.strftime("%Y-%m-%d")
                log_display['lot_date'] = log_display['lot_date'].dt.strftime("%Y-%m-%d")
                log_display = log_display.rename(columns={
                    'date':         'Sale Date',
                    'ticker':       'Ticker',
                    'sub_model':    'Sub-Model',
                    'lot_date':     'Purchase Date',
                    'holding_days': 'Days Held',
                    'rate_type':    'LT/ST',
                    'gain_%':       'Gain%',
                    'tax_%':        'Tax%',
                })
                st.dataframe(
                    log_display.style.format({'Gain%': '{:.3f}', 'Tax%': '{:.3f}'}),
                    use_container_width=True,
                    height=min(36 * (len(log_display) + 1) + 3, 600),
                )
            st.caption(
                "Gain% and Tax% expressed as % of portfolio value at period start.  "
                f"LTCG rate: {ltcg*100:.1f}%  |  STCG rate: {stcg*100:.1f}%"
            )

    # ── Positioning ────────────────────────────────────────────────────────────

    elif view == "Positioning":
        try:
            tier1, tier2 = compute_positioning()
        except Exception as e:
            st.error(f"Could not load positioning data: {e}")
            st.stop()

        TIER1_COLORS = {
            'Equity':       '#2563EB',
            'Fixed Income': '#F97316',
            'Cash':         '#94A3B8',
        }
        TIER2_COLORS = {
            'Fixed Income': '#F97316',
            'GV':           '#1D4ED8',
            'OMFL':         '#3B82F6',
            'Sector':       '#60A5FA',
            'T1F':          '#93C5FD',
            'SGA':          '#0EA5E9',
            'EAFE':         '#16A34A',
            'EM':           '#4ADE80',
            'SmallCap':     '#7C3AED',
            'Art':          '#A78BFA',
            'REIT':         '#DB2777',
            'Gold':         '#EAB308',
            'Cash':         '#94A3B8',
        }

        def stacked_bar(df, color_map, title, neutral_line=None):
            dates = df.index
            n     = len(dates)

            # Width of each bar = time until the next rebalance (ms for Plotly date axis)
            widths_ms, mid_x = [], []
            _fallback_ms = int(30 * 24 * 3600 * 1000)  # 30-day default for single-bar case
            for i in range(n):
                if i < n - 1:
                    dur_ms = int((dates[i + 1] - dates[i]).total_seconds() * 1000)
                elif n > 1:
                    dur_ms = int((dates[-1] - dates[-2]).total_seconds() * 1000)
                else:
                    dur_ms = _fallback_ms
                widths_ms.append(dur_ms)
                mid_x.append(dates[i] + pd.Timedelta(milliseconds=dur_ms // 2))

            fig = go.Figure()
            for col in df.columns:
                color = color_map.get(col, '#CCCCCC')
                fig.add_trace(go.Bar(
                    x=mid_x,
                    y=df[col].round(1),
                    width=widths_ms,
                    name=col,
                    marker_color=color,
                    customdata=dates.strftime('%b %d %Y'),
                    hovertemplate='%{customdata}<br>' + col + ': %{y:.1f}%<extra></extra>',
                ))
            if neutral_line is not None:
                fig.add_hline(
                    y=neutral_line, line_dash='dash', line_color='#374151', line_width=1.5,
                    annotation_text=f'Neutral ({neutral_line:.0f}%)',
                    annotation_position='top right',
                    annotation_font_size=11,
                )
            _bold = dict(color='black', size=13, family='Arial Black, Arial Bold, Arial')
            fig.update_layout(
                title=dict(text=f'<b>{title}</b>', font=dict(color='black', size=15, family='Arial')),
                barmode='stack',
                bargap=0,
                font=dict(color='black', family='Arial'),
                yaxis=dict(
                    title=dict(text='Weight (%)', font=_bold),
                    range=[0, 102], ticksuffix='%', tickfont=_bold,
                ),
                xaxis=dict(title='', tickfont=_bold),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,
                            font=dict(color='black', size=12, family='Arial')),
                margin=dict(t=80, b=40),
                height=420,
                plot_bgcolor='white',
                paper_bgcolor='white',
            )
            fig.update_xaxes(showgrid=False, tickfont=_bold)
            fig.update_yaxes(showgrid=True, gridcolor='#E5E7EB', tickfont=_bold)
            return fig

        st.subheader("Tier I — Equity vs Fixed Income")
        st.plotly_chart(
            stacked_bar(tier1, TIER1_COLORS,
                        "Portfolio Allocation Over Time",
                        neutral_line=60),
            use_container_width=True,
        )

        st.divider()

        st.subheader("Tier II — Equity Sub-Model Weights")
        tier2_eq = tier2.drop(columns=['Fixed Income'], errors='ignore')
        _eq_sums = tier2_eq.sum(axis=1).replace(0, float('nan'))
        tier2_eq = tier2_eq.div(_eq_sums, axis=0) * 100
        tier2_eq = tier2_eq.dropna(how='all')
        st.plotly_chart(
            stacked_bar(tier2_eq, TIER2_COLORS,
                        "Equity Sub-Model Allocation Over Time"),
            use_container_width=True,
        )
