"""Loading and analysis utilities for CC_Data.xlsx.

Sheets
------
  "Index Data"                 monthly returns, 33 tickers, Jan 1979 – present
  "Signals and Model Returns"  regime signals + model/benchmark returns, Jan 1988 – present
  "GIPS Returns"               GTAA PM composite model + benchmark returns (long format)
  "Bloomberg Model Returns"    PM / SGA / HNW model returns (grouped layout)

All loaders return monthly returns in percent (0.63 = 0.63%).
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from pathlib import Path

FILE_PATH = Path(__file__).parent / "CC_Data.xlsx"
MONTHS_PER_YEAR = 12


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_float(series: pd.Series) -> pd.Series:
    if pd.api.types.is_object_dtype(series):
        return pd.to_numeric(series.astype(str).str.rstrip("%"), errors="coerce")
    return pd.to_numeric(series, errors="coerce")


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_index_data(path: Path = FILE_PATH) -> pd.DataFrame:
    """'Index Data' sheet → MultiIndex columns (ticker, name), percent returns.

    Example
    -------
    idx = load_index_data()
    spx = idx.xs("S&P 500", level="name", axis=1).squeeze()
    eafe = idx.xs("GDDUEAFE Index", level="ticker", axis=1).squeeze()
    """
    raw = pd.read_excel(path, sheet_name="Index Data", header=None)
    # Row 0: Bloomberg tickers (col 0 = last-date reference)
    # Row 1: human-readable names
    # Row 2+: date in col 0, returns in cols 1+ (decimal form → × 100)
    tickers = raw.iloc[0, 1:].tolist()
    names   = raw.iloc[1, 1:].tolist()
    data    = raw.iloc[2:].copy()

    dates  = pd.to_datetime(data.iloc[:, 0], errors="coerce")
    values = data.iloc[:, 1:].apply(_to_float) * 100

    values.columns = pd.MultiIndex.from_arrays(
        [tickers, names], names=["ticker", "name"]
    )
    values.index = dates
    values.index.name = "date"
    return values.dropna(how="all")


def load_signals(path: Path = FILE_PATH) -> pd.DataFrame:
    """'Signals and Model Returns' sheet → date-indexed DataFrame.

    Signal columns (str):  Tier I Signal, GV Signal, EM Signal,
                           EAFE Signal, T1F Signal
    Return columns (float, %): AGG, ACWI, EAFE, EM, Gold, Growth, Value,
                                Momentum, Min Vol, R1
    Model columns (float, %):  Tier I Model, GV Model, EAFE Model, EM Model,
                                T1F Model, OMFL Model, Tier II, Market Risk
    """
    raw = pd.read_excel(path, sheet_name="Signals and Model Returns", header=0)
    date_col = raw.columns[0]
    raw.index = pd.to_datetime(raw[date_col], errors="coerce")
    raw.index.name = "date"
    raw = raw.drop(columns=[date_col])

    raw = raw.dropna(axis=1, how="all")
    raw = raw.loc[:, ~raw.columns.astype(str).str.startswith("Unnamed")]

    signal_cols = ["Tier I Signal", "GV Signal", "EM Signal", "EAFE Signal", "T1F Signal"]
    return_cols = [c for c in raw.columns if c not in signal_cols]
    raw[return_cols] = raw[return_cols].apply(_to_float) * 100

    return raw.dropna(how="all")


def load_gips(path: Path = FILE_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """'GIPS Returns' sheet → (model_returns, benchmarks), both in wide percent form.

    Each DataFrame has date as index and composite name as columns:
      GTAA PM Aggressive, GTAA PM Balanced, GTAA PM Conservative, GTAA PM Growth

    Example
    -------
    returns, benchmarks = load_gips()
    agg_excess = returns["GTAA PM Aggressive"] - benchmarks["GTAA PM Aggressive"]
    """
    raw = pd.read_excel(path, sheet_name="GIPS Returns", header=0)
    raw["Period"] = pd.to_datetime(raw["Period"], errors="coerce")
    raw = raw.dropna(subset=["Period", "Model Returns"])
    raw["Model Returns"] = _to_float(raw["Model Returns"]) * 100
    raw["Benchmark"]     = _to_float(raw["Benchmark"])     * 100

    model_ret = raw.pivot(index="Period", columns="Composite", values="Model Returns")
    benchmark = raw.pivot(index="Period", columns="Composite", values="Benchmark")

    model_ret.index.name = "date"
    benchmark.index.name = "date"
    model_ret.columns.name = None
    benchmark.columns.name = None

    return model_ret.sort_index(), benchmark.sort_index()


def load_bloomberg_models(path: Path = FILE_PATH) -> pd.DataFrame:
    """'Bloomberg Model Returns' sheet → date-indexed DataFrame of percent returns.

    Groups (each with its own date range):
      PM Core    : PM Aggressive, PM Growth, PM Balanced, PM Conservative  (Jan 2010)
      PM SGA     : PM Aggressive SGA, PM Growth SGA, PM Balanced SGA,
                   PM Conservative SGA                                      (Jan 2010)
      Other      : Appreciation, Growth and Income, Capital Preservation    (Feb 2021)
      HNW        : HNW Aggressive, HNW Growth, HNW Balanced,
                   HNW Conservative                                         (Feb 2025)

    Series with no data for a given month appear as NaN.
    """
    raw = pd.read_excel(path, sheet_name="Bloomberg Model Returns", header=None)

    # Each group: [date_col, price_col, return_col, price_col, return_col, ...]
    # Groups are separated by all-NaN spacer columns.
    # Row 0 = model names (NaN for price col, model name for each model pair start — actually
    #         model name sits on the price col and NaN on the return col).
    # Row 1 = tickers (same pattern).
    # Row 2+ = data.

    # Identify group starting columns: columns where row 0 contains a date value.
    group_date_cols = [
        c for c in raw.columns
        if pd.notna(raw.iloc[0, c]) and isinstance(raw.iloc[0, c], (pd.Timestamp,))
           or (pd.notna(raw.iloc[0, c]) and hasattr(raw.iloc[0, c], 'year'))
    ]

    # Manually define the group layout based on inspected structure:
    # (date_col, [(name, return_col), ...])
    GROUPS = [
        (0,  [("PM Aggressive", 2),   ("PM Growth", 4),
              ("PM Balanced",   6),   ("PM Conservative", 8)]),
        (10, [("PM Aggressive SGA", 12), ("PM Growth SGA", 14),
              ("PM Balanced SGA",   16), ("PM Conservative SGA", 18)]),
        (20, [("Appreciation", 22), ("Growth and Income", 24),
              ("Capital Preservation", 26)]),
        (28, [("HNW Aggressive", 30), ("HNW Growth", 32),
              ("HNW Balanced",   34), ("HNW Conservative", 36)]),
    ]

    frames = []
    for date_col, models in GROUPS:
        dates = pd.to_datetime(raw.iloc[2:, date_col], errors="coerce")
        group = pd.DataFrame(index=dates)
        group.index.name = "date"
        for name, ret_col in models:
            group[name] = _to_float(raw.iloc[2:, ret_col].values) * 100
        frames.append(group.dropna(how="all"))

    combined = pd.concat(frames, axis=1)
    combined = combined[~combined.index.isna()].sort_index()
    return combined


# ── Stats (all take monthly percent return series) ────────────────────────────

def annualized_return(returns: pd.Series, freq: int = MONTHS_PER_YEAR) -> float:
    """Geometric annualized return (%). 0.63 means 0.63%."""
    r = returns.dropna() / 100
    n = len(r)
    if n == 0:
        return np.nan
    return ((1 + r).prod() ** (freq / n) - 1) * 100


def annualized_vol(returns: pd.Series, freq: int = MONTHS_PER_YEAR) -> float:
    """Annualized standard deviation (%)."""
    return float(returns.dropna().std() * np.sqrt(freq))


def downside_deviation(
    returns: pd.Series, mar: float = 0.0, freq: int = MONTHS_PER_YEAR
) -> float:
    """Annualized downside deviation below a monthly MAR (%).

    mar: minimum acceptable monthly return in percent (default 0).
    """
    r = returns.dropna()
    shortfalls_sq = np.where(r < mar, (r - mar) ** 2, 0.0)
    return float(np.sqrt(shortfalls_sq.mean()) * np.sqrt(freq))


def tracking_error(
    portfolio: pd.Series, benchmark: pd.Series, freq: int = MONTHS_PER_YEAR
) -> float:
    """Annualized tracking error (std of monthly active returns, %)."""
    common = portfolio.dropna().index.intersection(benchmark.dropna().index)
    active = portfolio[common] - benchmark[common]
    return float(active.std() * np.sqrt(freq))


def excess_return(portfolio: pd.Series, benchmark: pd.Series) -> float:
    """Annualized excess return: ann(portfolio) − ann(benchmark), in %."""
    common = portfolio.dropna().index.intersection(benchmark.dropna().index)
    return annualized_return(portfolio[common]) - annualized_return(benchmark[common])


def information_ratio(portfolio: pd.Series, benchmark: pd.Series) -> float:
    """Information ratio: excess_return / tracking_error."""
    te = tracking_error(portfolio, benchmark)
    if te == 0:
        return np.nan
    return excess_return(portfolio, benchmark) / te


def sharpe_ratio(
    returns: pd.Series, rf: float = 0.0, freq: int = MONTHS_PER_YEAR
) -> float:
    """Annualized Sharpe ratio. rf = annualized risk-free rate in percent."""
    vol = annualized_vol(returns, freq)
    if vol == 0:
        return np.nan
    return (annualized_return(returns, freq) - rf) / vol


def sortino_ratio(
    returns: pd.Series, rf: float = 0.0, freq: int = MONTHS_PER_YEAR
) -> float:
    """Sortino ratio (excess return over rf divided by downside deviation)."""
    dd = downside_deviation(returns, mar=rf / freq, freq=freq)
    if dd == 0:
        return np.nan
    return (annualized_return(returns, freq) - rf) / dd


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative %, e.g. -34.5)."""
    r = returns.dropna() / 100
    cum  = (1 + r).cumprod()
    peak = cum.cummax()
    return float((cum / peak - 1).min() * 100)


def win_rate(returns: pd.Series) -> float:
    """Percentage of months with positive returns (0–100)."""
    r = returns.dropna()
    return float((r > 0).mean() * 100)


def cumulative_return(returns: pd.Series) -> pd.Series:
    """Cumulative wealth index, rebased to 100 at first observation."""
    r = returns.dropna() / 100
    return (1 + r).cumprod() * 100


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Time series of current drawdown from peak (negative %, e.g. -10.5)."""
    r = returns.dropna() / 100
    cum  = (1 + r).cumprod()
    peak = cum.cummax()
    return (cum / peak - 1) * 100


# ── Summary tables ────────────────────────────────────────────────────────────

def summary_stats(
    returns: "pd.DataFrame | pd.Series",
    benchmark: "pd.Series | None" = None,
    rf: float = 0.0,
) -> pd.DataFrame:
    """Comprehensive statistics table for one or more return series.

    Args:
        returns:   Series or DataFrame of monthly percent returns.
        benchmark: Optional benchmark series for relative stats
                   (tracking error, excess return, information ratio).
        rf:        Annualized risk-free rate in percent (default 0).

    Returns:
        DataFrame — one row per series, columns = stat names.
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()

    rows = {}
    for col in returns.columns:
        s = returns[col]
        row = {
            "Ann. Return (%)":  round(annualized_return(s), 2),
            "Ann. Std Dev (%)": round(annualized_vol(s), 2),
            "Sharpe":           round(sharpe_ratio(s, rf), 3),
            "Max Drawdown (%)": round(max_drawdown(s), 2),
            "Win Rate (%)":     round(win_rate(s), 1),
            "Months":           int(s.dropna().count()),
        }
        if benchmark is not None:
            common = s.dropna().index.intersection(benchmark.dropna().index)
            active = s[common] - benchmark[common]
            dd_active = downside_deviation(active, mar=0)
            exc_ret   = excess_return(s, benchmark)
            row["Excess Return (%)"]  = round(exc_ret, 2)
            row["Tracking Error (%)"] = round(tracking_error(s, benchmark), 2)
            row["Downside Dev (%)"]   = round(dd_active, 2)
            row["Sortino"]            = round(exc_ret / dd_active if dd_active != 0 else np.nan, 3)
            row["Info Ratio"]         = round(information_ratio(s, benchmark), 3)
        else:
            row["Downside Dev (%)"] = round(downside_deviation(s, mar=rf / MONTHS_PER_YEAR), 2)
            row["Sortino"]          = round(sortino_ratio(s, rf), 3)
        rows[col] = row

    if benchmark is not None:
        col_order = [
            "Ann. Return (%)", "Ann. Std Dev (%)", "Sharpe", "Max Drawdown (%)",
            "Win Rate (%)", "Months",
            "Excess Return (%)", "Tracking Error (%)", "Downside Dev (%)",
            "Sortino", "Info Ratio",
        ]
    else:
        col_order = [
            "Ann. Return (%)", "Ann. Std Dev (%)", "Downside Dev (%)",
            "Sharpe", "Sortino", "Max Drawdown (%)", "Win Rate (%)", "Months",
        ]

    return pd.DataFrame(rows).T[col_order]


# ── Charts ────────────────────────────────────────────────────────────────────

def ratio_chart(
    series_a: pd.Series,
    series_b: pd.Series,
    label_a: str | None = None,
    label_b: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Ratio of cumulative wealth of series_a to series_b (rebased to 1.0).

    A rising line means series_a is outperforming series_b on a cumulative basis.

    Example
    -------
    idx = load_index_data()
    growth = idx.xs("Russell 1000 Growth", level="name", axis=1).squeeze()
    value  = idx.xs("Russell 1000 Value",  level="name", axis=1).squeeze()
    ratio_chart(growth, value, "R1 Growth", "R1 Value")
    plt.show()
    """
    label_a = label_a or (series_a.name if series_a.name else "Series A")
    label_b = label_b or (series_b.name if series_b.name else "Series B")

    common = series_a.dropna().index.intersection(series_b.dropna().index)
    cum_a  = cumulative_return(series_a[common])
    cum_b  = cumulative_return(series_b[common])
    ratio  = cum_a / cum_b

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    ax.plot(ratio.index, ratio.values, linewidth=1.5)
    ax.axhline(1.0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_title(f"{label_a} / {label_b} — Cumulative Ratio")
    ax.set_ylabel("Ratio (rebased to 1.0 at start)")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("")
    return ax


def drawdown_chart(
    returns: "pd.DataFrame | pd.Series",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Drawdown-through-time chart (filled area, negative %).

    Example
    -------
    idx = load_index_data()
    spx = idx.xs("S&P 500", level="name", axis=1).squeeze()
    drawdown_chart(spx)
    plt.show()
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    for col in returns.columns:
        dd = drawdown_series(returns[col])
        ax.fill_between(dd.index, dd.values, 0, alpha=0.4, label=str(col))
        ax.plot(dd.index, dd.values, linewidth=0.8)

    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.axhline(0, color="black", linewidth=0.7)
    ax.grid(True, alpha=0.3)
    if len(returns.columns) > 1:
        ax.legend(fontsize=8)
    return ax


def relative_drawdown_chart(
    portfolio: "pd.DataFrame | pd.Series",
    benchmark: pd.Series,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Drawdown of cumulative relative performance (portfolio / benchmark).

    Shows how far each series has fallen from its peak relative to the benchmark.
    A value of -10% means the portfolio is 10% below its peak outperformance level.
    """
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


def rolling_excess_return_chart(
    portfolio: "pd.DataFrame | pd.Series",
    benchmark: pd.Series,
    windows: list[int] | None = None,
    label: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Rolling annualized excess-return chart for 1-year and 3-year windows.

    Computes rolling geometric excess return:
    ann(portfolio, w) − ann(benchmark, w) for each window w.

    Args:
        portfolio: Series or single-column DataFrame of monthly returns (%).
        benchmark: Benchmark series of monthly returns (%).
        windows:   Look-back windows in months (default [12, 36]).
        label:     Series label for the title/legend.

    Example
    -------
    sig = load_signals()
    rolling_excess_return_chart(sig["Tier I Model"], sig["R1"], label="Tier I Model")
    plt.show()
    """
    windows = windows or [12, 36]

    if isinstance(portfolio, pd.DataFrame):
        if portfolio.shape[1] != 1:
            raise ValueError("portfolio must be a Series or single-column DataFrame")
        portfolio = portfolio.iloc[:, 0]

    label = label or (portfolio.name if portfolio.name else "Portfolio")
    common = portfolio.dropna().index.intersection(benchmark.dropna().index)
    p = portfolio[common]
    b = benchmark[common]

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, w in enumerate(windows):
        roll_p = p.rolling(w).apply(
            lambda x: annualized_return(pd.Series(x)), raw=False
        )
        roll_b = b.rolling(w).apply(
            lambda x: annualized_return(pd.Series(x)), raw=False
        )
        roll_excess = roll_p - roll_b
        years = w // MONTHS_PER_YEAR
        lbl = f"{years}yr" if w % MONTHS_PER_YEAR == 0 else f"{w}mo"
        ax.plot(roll_excess.index, roll_excess.values,
                linewidth=1.3, label=lbl, color=colors[i % len(colors)])

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title(f"{label} — Rolling Excess Return vs Benchmark")
    ax.set_ylabel("Annualized Excess Return (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    return ax


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading Index Data...")
    idx = load_index_data()
    print(f"  {idx.shape[0]} rows, {idx.shape[1]} series  "
          f"({idx.index.min().date()} to {idx.index.max().date()})")

    print("Loading Signals...")
    sig = load_signals()
    print(f"  {sig.shape[0]} rows, columns: {sig.columns.tolist()}")

    print("Loading GIPS Returns...")
    gips_ret, gips_bench = load_gips()
    print(f"  {gips_ret.shape[0]} rows, composites: {gips_ret.columns.tolist()}")

    print("Loading Bloomberg Models...")
    bbg = load_bloomberg_models()
    print(f"  {bbg.shape[0]} rows, models: {bbg.columns.tolist()}")

    # Sample stats: S&P 500 vs MSCI EAFE
    print()
    spx  = idx.xs("S&P 500",   level="name", axis=1).squeeze().dropna()
    eafe = idx.xs("MSCI EAFE", level="name", axis=1).squeeze().dropna()
    print("S&P 500 vs MSCI EAFE summary:")
    print(summary_stats(pd.concat([spx, eafe], axis=1, keys=["S&P 500", "MSCI EAFE"])).to_string())

    # Sample: Tier I Model vs R1 benchmark
    print()
    tier1 = sig["Tier I Model"].dropna()
    r1    = sig["R1"].dropna()
    print("Tier I Model vs Russell 1000:")
    print(summary_stats(tier1.to_frame("Tier I Model"), benchmark=r1).to_string())
