"""
attribution.py — Two-level Brinson attribution for CC Analytics.

Level 1 (Tier I): equity/bond timing vs. neutral weight benchmark
Level 2 (within equity): geographic allocation (EAFE, EM) + sub-model contributions vs. ACWI

Attribution effects are computed daily (to handle intramonth rebalances) and linked
geometrically using the Carino (1999) method before aggregation. All effects sum
exactly to the geometric excess return; no residual term is needed.
"""

import functools

import pandas as pd
import numpy as np

FILE = 'CC_Data.xlsx'

# Neutral equity weight by portfolio type
NEUTRAL_EQUITY = {
    'Aggressive':   0.80,
    'Growth':       0.70,
    'Balanced':     0.60,
    'Conservative': 0.40,
}

# Sub-models and their asset class
EQUITY_SUBMODELS  = {'GV', 'EAFE', 'EM', 'Sector', 'T1F', 'OMFL', 'SGA', 'SmallCap', 'Art', 'REIT', 'Gold'}
GEO_SUBMODELS     = {'EAFE', 'EM'}       # have ACWI benchmark weights; get geo-allocation split
R1000_SUBMODELS   = {'GV', 'OMFL', 'Sector'}  # US-only models decomposed vs Russell 1000 (IWB)

# Benchmark tickers (must be present in ETF_Returns)
BM_EQUITY = 'ACWI'
BM_BOND   = 'MUB'
BM_EAFE   = 'IEFA'
BM_EM     = 'IEMG'
BM_R1000  = 'IWB'   # iShares Russell 1000 — used for GV US-market vs style decomposition


def _bond_bm(rebal_sheet):
    """Bond benchmark: AGG for TE (tax-exempt) model, MUB otherwise."""
    return 'AGG' if rebal_sheet == 'TE Rebalances' else BM_BOND


# ── Loaders ───────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=4)
def load_etf_returns(path=FILE):
    """
    Parse Bloomberg paired-column daily returns sheet.
    Each ticker occupies two columns: (date_col, return_col).
    Returns a date-indexed DataFrame with tickers as columns (returns in %).
    USD (cash) column is added here with 0% return.
    """
    raw = pd.read_excel(path, sheet_name='ETF_Returns', header=0)
    raw = raw.iloc[:, 1:]   # drop leading empty column

    cols = list(raw.columns)
    series_list = []

    for i in range(0, len(cols) - 1, 2):
        ticker = str(cols[i]).strip()
        if ticker.startswith('Unnamed'):
            continue

        dates_raw = raw.iloc[:, i]
        vals_raw  = raw.iloc[:, i + 1]
        mask = dates_raw.notna() & vals_raw.notna()

        try:
            dates = pd.to_datetime(dates_raw[mask], errors='coerce')
            vals  = pd.to_numeric(vals_raw[mask], errors='coerce')
        except Exception:
            continue

        # Drop rows where the date parsed as epoch (numeric 0 in Excel → 1970-01-01)
        valid = dates > pd.Timestamp('2000-01-01')
        dates = dates[valid]
        vals  = vals[valid.values]

        s = pd.Series(vals.values, index=dates.values, name=ticker)
        series_list.append(s)

    df = pd.concat(series_list, axis=1)
    df.index = pd.to_datetime(df.index)
    df.index.name = 'date'
    df = df.sort_index()
    df['USD'] = 0.0
    return df


@functools.lru_cache(maxsize=8)
def load_rebalances(path=FILE, sheet_name='Rebalances'):
    """
    Load a rebalance sheet.
    Normalises column names, asset_class values, and converts Weight_pct to decimals.
    """
    df = pd.read_excel(path, sheet_name=sheet_name)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df['date']       = pd.to_datetime(df['date'])
    df['weight_pct'] = df['weight_pct'] / 100.0
    df['asset_class'] = (df['asset_class'].str.strip()
                                          .str.replace(' ', '')
                                          .replace({'FixedIncome': 'FixedIncome'}))
    df['sub_model']  = (df['sub_model'].str.strip()
                                       .str.replace(' ', ''))
    return df.sort_values('date').reset_index(drop=True)


def load_acwi_weights(path=FILE):
    """
    Load ACWI geographic weights (EAFE, EM, US) as decimals.
    Returns a date-indexed DataFrame, monthly.
    """
    df = pd.read_excel(path, sheet_name='ACWI Weights')
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'])
    return df.set_index('Date').sort_index()


# ── Weight helpers ────────────────────────────────────────────────────────────

def build_daily_weights(rebalances, trading_dates):
    """
    Forward-fill rebalance weights to every date in trading_dates.
    Weights take effect the trading day AFTER the rebalance date (trades execute at close).
    Returns DataFrame: index=date, columns=ticker, values=decimal weight.
    """
    piv = (rebalances
           .pivot_table(index='date', columns='ticker',
                        values='weight_pct', aggfunc='sum')
           .fillna(0))

    all_dates = piv.index.union(pd.DatetimeIndex(trading_dates))
    daily = piv.reindex(all_dates).sort_index().ffill()
    daily = daily.reindex(trading_dates).fillna(0)
    # Shift by 1 trading day: rebalance date uses pre-trade weights; new weights active T+1
    return daily.shift(1).fillna(0)


def get_ticker_meta(rebalances):
    """
    Return {ticker: {'sub_model': ..., 'asset_class': ...}} from the most
    recent rebalance entry for each ticker.
    """
    latest = (rebalances.sort_values('date')
                        .drop_duplicates('ticker', keep='last'))
    return latest.set_index('ticker')[['sub_model', 'asset_class']].to_dict('index')


# ── Core attribution ──────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=8)
def _build_daily_attribution(neutral_equity=0.60, path=FILE,
                              rebal_sheet='Rebalances', equity_only=False):
    """
    Internal: build the full daily attribution DataFrame.
    All public functions call this and then aggregate as needed.

    equity_only=True: normalise weights by total equity weight and use ACWI as the
    benchmark; Tier I and Fixed Income effects are suppressed (set to 0).

    Returns a DataFrame indexed by trading date with columns:
        r_port, r_bm                       — daily portfolio / benchmark returns (%)
        tier1, fi_selection                — Level 1 effects (0 in equity_only mode)
        <sub_model>                        — equity sub-model contributions
        EAFE_bm/alloc/sel                  — secondary EAFE decomposition
        EM_bm/alloc/sel                    — secondary EM decomposition
    """
    rebalances   = load_rebalances(path, rebal_sheet)
    etf_returns  = load_etf_returns(path)
    acwi_weights = load_acwi_weights(path)

    bond_bm = _bond_bm(rebal_sheet)
    for bm in [BM_EQUITY, bond_bm, BM_EAFE, BM_EM]:
        if bm not in etf_returns.columns:
            raise ValueError(f"Benchmark ticker '{bm}' not found in ETF_Returns sheet.")

    trading_dates = etf_returns.index
    daily_weights = build_daily_weights(rebalances, trading_dates)
    meta          = get_ticker_meta(rebalances)

    acwi_daily = (acwi_weights
                  .reindex(acwi_weights.index.union(trading_dates))
                  .sort_index().ffill()
                  .reindex(trading_dates))

    tickers_held = daily_weights.columns.intersection(etf_returns.columns)
    missing = set(daily_weights.columns) - set(etf_returns.columns)
    if missing:
        print(f"WARNING: no return data for: {missing} — excluded from attribution")

    W = daily_weights[tickers_held]
    # Fill individual missing returns with 0 so weighting and attribution stay consistent.
    # A ticker with no data on a given day is treated as earning 0% (holds value).
    R = etf_returns[tickers_held].reindex(trading_dates).fillna(0)

    is_equity  = pd.Series({t: meta.get(t, {}).get('asset_class', '') == 'Equity'
                             for t in tickers_held})
    eq_tickers = is_equity[is_equity].index.tolist()
    fi_tickers = is_equity[~is_equity].index.tolist()

    r_acwi = etf_returns[BM_EQUITY].reindex(trading_dates)
    r_agg  = etf_returns[bond_bm].reindex(trading_dates)
    r_iefa = etf_returns[BM_EAFE].reindex(trading_dates)
    r_iemg = etf_returns[BM_EM].reindex(trading_dates)
    r_iwb  = (etf_returns[BM_R1000].reindex(trading_dates)
              if BM_R1000 in etf_returns.columns else None)
    w_eq = W[eq_tickers].sum(axis=1)

    def weighted_return(tickers_):
        total_w = W[tickers_].sum(axis=1).replace(0, np.nan)
        return (W[tickers_] * R[tickers_]).sum(axis=1) / total_w

    if equity_only:
        # Portfolio = equity holdings only (weights normalised to sum to 1 within equity)
        w_eq_total   = w_eq.replace(0, np.nan)
        r_port       = ((W[eq_tickers] * R[eq_tickers]).sum(axis=1) / w_eq_total).fillna(0)
        r_bm         = r_acwi
        tier1        = pd.Series(0.0, index=trading_dates)
        fi_selection = pd.Series(0.0, index=trading_dates)
        w_eq_scale   = (1.0 / w_eq_total).fillna(0)  # per-day normalisation factor
        neutral_factor = 1.0  # neutral EAFE/EM weight = ACWI's own weight
    else:
        r_port         = (W * R).sum(axis=1)
        r_bm           = neutral_equity * r_acwi + (1 - neutral_equity) * r_agg
        tier1          = (w_eq - neutral_equity) * (r_acwi - r_agg)
        fi_selection   = (W[fi_tickers] * (R[fi_tickers].subtract(r_agg, axis=0))).sum(axis=1)
        w_eq_scale     = 1.0
        neutral_factor = neutral_equity

    sub_contribs = {}
    for sm in sorted(EQUITY_SUBMODELS):
        sm_tickers = [t for t in eq_tickers
                      if meta.get(t, {}).get('sub_model', '') == sm]
        if not sm_tickers:
            continue

        w_sm     = W[sm_tickers].sum(axis=1)
        w_sm_eff = w_sm * w_eq_scale          # equity-normalised in equity_only, raw otherwise
        r_sm     = weighted_return(sm_tickers).fillna(0)
        # Contribution (primary reconciling effect): w × (r_sm − r_ACWI)
        sub_contribs[sm] = w_sm_eff * (r_sm - r_acwi)

        if sm == 'EAFE':
            neutral_eafe = neutral_factor * acwi_daily['EAFE']
            # Three-term decomposition — sums exactly to EAFE contribution:
            #   bm_exp  = neutral × (IEFA − ACWI)          : drag/boost from benchmark EAFE weight
            #   alloc   = (actual − neutral) × (IEFA − ACWI): value of active OW/UW decision
            #   sel     = actual × (ETF − IEFA)             : ETF alpha vs benchmark index
            sub_contribs['EAFE_bm']    =  neutral_eafe               * (r_iefa - r_acwi)
            sub_contribs['EAFE_alloc'] = (w_sm_eff - neutral_eafe)   * (r_iefa - r_acwi)
            sub_contribs['EAFE_sel']   =  w_sm_eff                   * (r_sm   - r_iefa)
        elif sm == 'EM':
            neutral_em = neutral_factor * acwi_daily['EM']
            sub_contribs['EM_bm']      =  neutral_em                 * (r_iemg - r_acwi)
            sub_contribs['EM_alloc']   = (w_sm_eff - neutral_em)     * (r_iemg - r_acwi)
            sub_contribs['EM_sel']     =  w_sm_eff                   * (r_sm   - r_iemg)
        elif sm in R1000_SUBMODELS and r_iwb is not None:
            # Two-term decomposition vs Russell 1000:
            #   us_mkt = w × (IWB − ACWI) : US market exposure vs global benchmark
            #   style  = w × (model − IWB): factor/style alpha vs plain R1000
            sub_contribs[f'{sm}_us_mkt'] = w_sm_eff * (r_iwb - r_acwi)
            sub_contribs[f'{sm}_style']  = w_sm_eff * (r_sm  - r_iwb)

    df = pd.DataFrame({
        'r_port':       r_port,
        'r_bm':         r_bm,
        'tier1':        tier1,
        'fi_selection': fi_selection,
        **sub_contribs,
    })
    # Drop any days where benchmark return is missing (e.g. data gaps for recent dates)
    return df.dropna(subset=['r_bm'])


def _apply_carino_geometric(daily):
    """
    Scale daily attribution effects using a modified Carino (1999) coefficient
    so that their sum equals the geometric excess return over the period.

    Standard Carino uses K = log_excess / arith_excess  → effects sum to arith excess.
    We use K_geo = log_excess / geo_excess               → effects sum to geo excess.

    Daily scaling factor:  k_t / K_geo  where
        k_t   = [ln(1+r_P_t/100) - ln(1+r_B_t/100)] / ((r_P_t - r_B_t)/100)
        K_geo = [ln(1+R_P/100)   - ln(1+R_B/100)]   / geo_excess_decimal

    L'Hôpital limit applies on days where portfolio ≈ benchmark: k_t → 1/(1+r_P_t/100).
    """
    r_port = daily['r_port']  # %
    r_bm   = daily['r_bm']    # %

    R_P = ((1 + r_port / 100).prod() - 1) * 100
    R_B = ((1 + r_bm   / 100).prod() - 1) * 100

    geo_excess_dec = (1 + R_P / 100) / (1 + R_B / 100) - 1
    log_excess_dec = np.log(1 + R_P / 100) - np.log(1 + R_B / 100)

    if abs(geo_excess_dec) < 1e-10:
        return daily  # no excess to link to; return unscaled

    K_geo = log_excess_dec / geo_excess_dec

    daily_diff_dec = (r_port - r_bm) / 100
    log_diff       = np.log(1 + r_port / 100) - np.log(1 + r_bm / 100)

    k_t = np.where(
        daily_diff_dec.abs() < 1e-10,
        1.0 / (1 + r_port / 100),
        log_diff / daily_diff_dec,
    )
    scale = pd.Series(k_t / K_geo, index=daily.index)

    result = daily.copy()
    for col in daily.columns:
        if col not in ('r_port', 'r_bm'):
            result[col] = daily[col] * scale
    return result


def _summarise(daily):
    """
    Collapse a daily attribution DataFrame into a single summary Series.
    Portfolio and benchmark returns are compounded geometrically.
    Attribution effects are Carino-linked so they sum to geometric excess.
    """
    def compound(s):
        return ((1 + s / 100).prod() - 1) * 100

    linked      = _apply_carino_geometric(daily)
    effect_cols = [c for c in daily.columns if c not in ('r_port', 'r_bm')]
    result = linked[effect_cols].sum().to_dict()
    result['portfolio_return'] = compound(daily['r_port'])
    result['benchmark_return'] = compound(daily['r_bm'])

    primary = ['tier1', 'fi_selection'] + [c for c in effect_cols if c in EQUITY_SUBMODELS]
    result['excess_return'] = sum(result[c] for c in primary)

    secondary = (['EAFE_bm', 'EAFE_alloc', 'EAFE_sel',
                  'EM_bm',   'EM_alloc',   'EM_sel']
                 + [f'{sm}_us_mkt' for sm in sorted(R1000_SUBMODELS)]
                 + [f'{sm}_style'  for sm in sorted(R1000_SUBMODELS)])
    col_order = (['portfolio_return', 'benchmark_return', 'excess_return',
                  'tier1', 'fi_selection']
                 + sorted(EQUITY_SUBMODELS & set(result))
                 + [c for c in secondary if c in result])
    return pd.Series({k: result[k] for k in col_order if k in result})


def compute_monthly_attribution(neutral_equity=0.60, path=FILE,
                                rebal_sheet='Rebalances', equity_only=False):
    """
    Compute two-level attribution aggregated to calendar months.
    Carino geometric linking is applied within each month so each month's effects
    sum to that month's geometric excess return.
    Returns a DataFrame indexed by month-end date.
    """
    daily = _build_daily_attribution(neutral_equity, path, rebal_sheet, equity_only)

    def compound(s):
        return ((1 + s / 100).prod() - 1) * 100

    def _month_carino_sum(g):
        linked = _apply_carino_geometric(g)
        effect_cols = [c for c in g.columns if c not in ('r_port', 'r_bm')]
        return linked[effect_cols].sum()

    monthly_ret = (daily[['r_port', 'r_bm']]
                   .groupby(pd.Grouper(freq='ME'))
                   .agg(portfolio_return=('r_port', compound),
                        benchmark_return=('r_bm',   compound)))

    monthly_eff = daily.groupby(pd.Grouper(freq='ME')).apply(_month_carino_sum)

    monthly = pd.concat([monthly_ret, monthly_eff], axis=1)

    effect_cols = [c for c in daily.columns if c not in ('r_port', 'r_bm')]
    primary = ['tier1', 'fi_selection'] + [c for c in effect_cols if c in EQUITY_SUBMODELS]
    monthly['excess_return'] = monthly[primary].sum(axis=1)

    monthly.index.name = 'date'
    return monthly


def _period_slice(daily, start, end):
    """Return daily rows for the period (start, end] — start date excluded."""
    return daily[(daily.index > pd.Timestamp(start)) & (daily.index <= pd.Timestamp(end))]


def compute_period_attribution(start, end, neutral_equity=0.60, path=FILE,
                               rebal_sheet='Rebalances', equity_only=False):
    """
    Compute attribution for an exact date range.

    Parameters
    ----------
    start : str or date-like  e.g. '2025-06-30'
    end   : str or date-like  e.g. '2026-05-29'

    Returns a Series with portfolio_return, benchmark_return, excess_return,
    and all attribution effects over the period.
    """
    daily  = _build_daily_attribution(neutral_equity, path, rebal_sheet, equity_only)
    period = _period_slice(daily, start, end)
    if period.empty:
        raise ValueError(f"No trading days found between {start} and {end}.")
    result = _summarise(period)
    result.name = f"{pd.Timestamp(start).date()} to {pd.Timestamp(end).date()}"
    return result


def compute_attribution_table(start, end, neutral_equity=0.60, path=FILE,
                              rebal_sheet='Rebalances', equity_only=False):
    """
    Two-level Brinson attribution table for a date range.
    Carino geometric linking is applied so all effects sum to the geometric excess.

    Columns: Allocation (Passive) | Allocation (Active) | Selection | Total

    Level 1:
        Tier I        — Allocation (Active) = (w_eq - neutral) x (ACWI - MUB)
        Fixed Income  — Selection           = sum w_fi x (r_fi - MUB)

    Level 2 — EAFE / EM (three terms sum exactly to Total):
        Allocation (Passive) = neutral_wt x (bench - ACWI)         : benchmark's geo weight
        Allocation (Active)  = (actual - neutral) x (bench - ACWI) : active OW/UW decision
        Selection            = actual_wt x (ETF - bench)           : ETF vs index
        Total                = actual_wt x (ETF - ACWI)            : net contribution

    Level 2 — US allocation (aggregate R1000 exposure):
        Allocation (Passive) = neutral_US x (IWB - ACWI)
        Allocation (Active)  = (actual_US - neutral_US) x (IWB - ACWI)
        Total                = Passive + Active

    Level 2 — GV / OMFL / Sector (style selection within US):
        Selection = Total = w x (r_model - IWB);  allocation columns blank

    Level 2 — off-benchmark models (SGA, T1F, SmallCap, etc.):
        Selection = Total = w x (r - ACWI);  allocation columns blank
    """
    daily  = _build_daily_attribution(neutral_equity, path, rebal_sheet, equity_only)
    period = _period_slice(daily, start, end)
    if period.empty:
        raise ValueError(f"No trading days found between {start} and {end}.")

    def compound(s):
        return ((1 + s / 100).prod() - 1) * 100

    port_ret   = compound(period['r_port'])
    bm_ret     = compound(period['r_bm'])
    geo_excess = ((1 + port_ret / 100) / (1 + bm_ret / 100) - 1) * 100

    # Carino geometric linking: scale daily effects so they sum to geo_excess
    linked = _apply_carino_geometric(period)
    eff    = linked.drop(columns=['r_port', 'r_bm']).sum()

    COLS = ['Allocation (Passive)', 'Allocation (Active)', 'Selection', 'Total']
    NaN  = float('nan')

    def row(**kw):
        return {c: kw.get(c, NaN) for c in COLS}

    rows = {}

    # ── Summary ──
    rows['Portfolio Return'] = row(Total=port_ret)
    rows['Benchmark Return'] = row(Total=bm_ret)
    rows['Excess Return']    = row(Total=geo_excess)

    # ── Level 1 (suppressed in equity-only mode) ──
    if not equity_only:
        rows['Tier I (equity/bond)'] = row(**{'Allocation (Active)': eff.get('tier1', 0),
                                              'Total': eff.get('tier1', 0)})
        rows['Fixed Income'] = row(Selection=eff.get('fi_selection', 0),
                                   Total=eff.get('fi_selection', 0))

    # ── Level 2 — geographic models ──
    for sm, bm_col, a_col, s_col in [
        ('EAFE', 'EAFE_bm', 'EAFE_alloc', 'EAFE_sel'),
        ('EM',   'EM_bm',   'EM_alloc',   'EM_sel'),
    ]:
        if sm not in eff:
            continue
        rows[sm] = row(**{
            'Allocation (Passive)': eff.get(bm_col, NaN),
            'Allocation (Active)':  eff.get(a_col,  NaN),
            'Selection':            eff.get(s_col,  NaN),
            'Total':                eff.get(sm, 0),
        })

    # ── Level 2 — US / off-benchmark models ──
    for sm in sorted(EQUITY_SUBMODELS - GEO_SUBMODELS):
        if sm not in eff:
            continue
        if sm in R1000_SUBMODELS and f'{sm}_us_mkt' in eff:
            rows[sm] = row(**{
                'Allocation (Passive)': eff.get(f'{sm}_us_mkt', NaN),
                'Selection':            eff.get(f'{sm}_style',  NaN),
                'Total':                eff.get(sm, 0),
            })
        else:
            rows[sm] = row(Selection=eff.get(sm, 0), Total=eff.get(sm, 0))

    tbl = pd.DataFrame.from_dict(rows, orient='index', columns=COLS)
    tbl.index.name = 'Effect'

    # ── Position stats: Wt%, Bm Wt%, Ret%, Bm Ret%, Contrib%, Bm Contrib% ──────

    bond_bm   = _bond_bm(rebal_sheet)
    STAT_COLS = ['Wt%', 'Bm Wt%', 'Ret%', 'Bm Ret%', 'Contrib%', 'Bm Contrib%']
    stats = pd.DataFrame({c: NaN for c in STAT_COLS}, index=tbl.index)

    rebalances_  = load_rebalances(path, rebal_sheet)
    etf_returns_ = load_etf_returns(path)
    meta_        = get_ticker_meta(rebalances_)
    daily_w_     = build_daily_weights(rebalances_, etf_returns_.index)
    tickers_h    = daily_w_.columns.intersection(etf_returns_.columns)

    W_p = daily_w_[tickers_h].reindex(period.index)
    R_p = etf_returns_[tickers_h].reindex(period.index).fillna(0)

    is_eq_ = pd.Series({t: meta_.get(t, {}).get('asset_class', '') == 'Equity'
                        for t in tickers_h})
    eq_tks = is_eq_[is_eq_].index.tolist()
    fi_tks = is_eq_[~is_eq_].index.tolist()

    if equity_only:
        w_eq_tot_ = W_p[eq_tks].sum(axis=1).replace(0, np.nan)
        W_p = W_p.copy()
        for t in eq_tks:
            W_p[t] = W_p[t].div(w_eq_tot_)

    ra_ = etf_returns_[BM_EQUITY].reindex(period.index)
    rb_ = etf_returns_[bond_bm].reindex(period.index)
    re_ = etf_returns_[BM_EAFE].reindex(period.index)
    rm_ = etf_returns_[BM_EM].reindex(period.index)
    rw_ = (etf_returns_[BM_R1000].reindex(period.index)
           if BM_R1000 in etf_returns_.columns else None)

    acwi_w_ = load_acwi_weights(path)
    acwi_p_ = (acwi_w_.reindex(acwi_w_.index.union(period.index))
                .sort_index().ffill().reindex(period.index))

    nf = 1.0 if equity_only else neutral_equity

    def _cpd_(s):
        return round(float(((1 + s / 100).prod() - 1) * 100), 2)

    def _avg_wt_(tks):
        return round(float(W_p[tks].sum(axis=1).mean()) * 100, 2) if tks else NaN

    def _sm_ret_(tks):
        if not tks:
            return NaN
        w = W_p[tks].sum(axis=1).replace(0, np.nan)
        r = (W_p[tks] * R_p[tks]).sum(axis=1)
        return round(float(((1 + (r / w).fillna(0) / 100).prod() - 1) * 100), 2)

    # Carino geometric scale: daily k_t/K so contributions sum to geometric portfolio/bm return.
    def _carino_scale(r_daily, R_period):
        R_dec  = R_period / 100
        K      = np.log(1 + R_dec) / R_dec if abs(R_dec) > 1e-10 else 1.0
        r_frac = r_daily / 100
        kt     = np.where(r_frac.abs() < 1e-10,
                          1.0 / (1 + r_frac),
                          np.log(1 + r_frac) / r_frac)
        return pd.Series(kt / K, index=r_daily.index)

    scale_port_ = _carino_scale(period['r_port'], port_ret)
    scale_bm_   = _carino_scale(period['r_bm'],   bm_ret)

    def _pc_(tks):
        if not tks:
            return NaN
        return round(float(((W_p[tks] * R_p[tks]).sum(axis=1) * scale_port_).sum()), 3)

    def _bc_(bm_wt_s, r_s):
        return round(float((bm_wt_s * r_s * scale_bm_).sum()), 3)

    # Tier I / FI (standard mode only)
    if not equity_only:
        w_eq_ = W_p[eq_tks].sum(axis=1) if eq_tks else pd.Series(0.0, index=period.index)
        if 'Tier I (equity/bond)' in stats.index:
            stats.loc['Tier I (equity/bond)', 'Wt%']    = round(float(w_eq_.mean()) * 100, 2)
            stats.loc['Tier I (equity/bond)', 'Bm Wt%'] = round(neutral_equity * 100, 2)
        if 'Fixed Income' in stats.index:
            fi_bm_wt = pd.Series(1 - neutral_equity, index=period.index)
            stats.loc['Fixed Income', 'Wt%']        = _avg_wt_(fi_tks)
            stats.loc['Fixed Income', 'Bm Wt%']     = round((1 - neutral_equity) * 100, 2)
            stats.loc['Fixed Income', 'Ret%']        = _sm_ret_(fi_tks)
            stats.loc['Fixed Income', 'Bm Ret%']     = _cpd_(rb_)
            stats.loc['Fixed Income', 'Contrib%']    = _pc_(fi_tks)
            stats.loc['Fixed Income', 'Bm Contrib%'] = _bc_(fi_bm_wt, rb_)

    # EAFE
    if 'EAFE' in stats.index:
        eafe_tks = [t for t in eq_tks if meta_.get(t, {}).get('sub_model', '') == 'EAFE']
        bm_wt_e  = nf * acwi_p_['EAFE']
        stats.loc['EAFE', 'Wt%']        = _avg_wt_(eafe_tks)
        stats.loc['EAFE', 'Ret%']        = _cpd_(re_)
        stats.loc['EAFE', 'Bm Ret%']     = _cpd_(re_)
        stats.loc['EAFE', 'Contrib%']    = _pc_(eafe_tks)
        stats.loc['EAFE', 'Bm Contrib%'] = _bc_(bm_wt_e, re_)

    # EM
    if 'EM' in stats.index:
        em_tks  = [t for t in eq_tks if meta_.get(t, {}).get('sub_model', '') == 'EM']
        bm_wt_m = nf * acwi_p_['EM']
        stats.loc['EM', 'Wt%']        = _avg_wt_(em_tks)
        stats.loc['EM', 'Ret%']        = _cpd_(rm_)
        stats.loc['EM', 'Bm Ret%']     = _cpd_(rm_)
        stats.loc['EM', 'Contrib%']    = _pc_(em_tks)
        stats.loc['EM', 'Bm Contrib%'] = _bc_(bm_wt_m, rm_)

    # R1000 models — no individual benchmark allocation
    for sm in sorted(R1000_SUBMODELS):
        if sm not in stats.index:
            continue
        sm_tks = [t for t in eq_tks if meta_.get(t, {}).get('sub_model', '') == sm]
        stats.loc[sm, 'Wt%']      = _avg_wt_(sm_tks)
        stats.loc[sm, 'Ret%']     = _sm_ret_(sm_tks)
        stats.loc[sm, 'Bm Ret%']  = _cpd_(rw_) if rw_ is not None else NaN
        stats.loc[sm, 'Contrib%'] = _pc_(sm_tks)

    # Off-benchmark equity models
    for sm in sorted(EQUITY_SUBMODELS - GEO_SUBMODELS - R1000_SUBMODELS):
        if sm not in stats.index:
            continue
        sm_tks = [t for t in eq_tks if meta_.get(t, {}).get('sub_model', '') == sm]
        stats.loc[sm, 'Wt%']        = _avg_wt_(sm_tks)
        stats.loc[sm, 'Ret%']        = _sm_ret_(sm_tks)
        stats.loc[sm, 'Bm Ret%']     = _cpd_(ra_)
        stats.loc[sm, 'Contrib%']    = _pc_(sm_tks)
        stats.loc[sm, 'Bm Contrib%'] = 0.0

    # Summary rows
    if 'Portfolio Return' in stats.index:
        stats.loc['Portfolio Return', 'Ret%'] = round(port_ret, 2)
    if 'Benchmark Return' in stats.index:
        stats.loc['Benchmark Return', 'Bm Ret%'] = round(bm_ret, 2)

    # ── Per-FI-ticker rows (bond drill-down) ────────────────────────────────────
    if not equity_only:
        # Carino scale for excess-return attribution (same formula as _apply_carino_geometric)
        geo_exc = (1 + port_ret / 100) / (1 + bm_ret / 100) - 1
        log_exc = np.log(1 + port_ret / 100) - np.log(1 + bm_ret / 100)
        if abs(geo_exc) > 1e-10:
            K_exc   = log_exc / geo_exc
            diff    = (period['r_port'] - period['r_bm']) / 100
            log_d   = np.log(1 + period['r_port'] / 100) - np.log(1 + period['r_bm'] / 100)
            k_exc   = np.where(diff.abs() < 1e-10, 1.0 / (1 + period['r_port'] / 100), log_d / diff)
            scale_exc_ = pd.Series(k_exc / K_exc, index=period.index)
        else:
            scale_exc_ = pd.Series(1.0, index=period.index)

        bm_ret_pct = _cpd_(rb_)

        for t in fi_tks:
            if t == 'USD':
                continue
            avg_w = _avg_wt_([t])
            if not avg_w:           # skip tickers with no weight in this period
                continue
            daily_sel  = (W_p[[t]] * R_p[[t]].subtract(rb_, axis=0)).sum(axis=1)
            ticker_sel = round(float((daily_sel * scale_exc_).sum()), 3)

            fi_stat = {c: NaN for c in STAT_COLS}
            fi_stat['Wt%']      = avg_w
            fi_stat['Ret%']     = _sm_ret_([t])
            fi_stat['Bm Ret%']  = bm_ret_pct
            fi_stat['Contrib%'] = _pc_([t])

            fi_attr = {c: NaN for c in COLS}
            fi_attr['Selection'] = ticker_sel
            fi_attr['Total']     = ticker_sel

            stats = pd.concat([stats, pd.DataFrame([fi_stat], index=[t], columns=STAT_COLS)])
            tbl   = pd.concat([tbl,   pd.DataFrame([fi_attr], index=[t], columns=COLS)])

    # ── Total Stock / Total Bond aggregate rows ─────────────────────────────────
    _skip       = {'Portfolio Return', 'Benchmark Return', 'Excess Return',
                   'Tier I (equity/bond)', 'Fixed Income'}
    eq_sub_rows = [k for k in tbl.index if k not in _skip]

    ts_attr  = dict(tbl.loc[eq_sub_rows].sum(skipna=True))
    ts_stats = {c: NaN for c in STAT_COLS}
    ts_stats.update({
        'Wt%':         _avg_wt_(eq_tks),
        'Bm Wt%':      round(nf * 100, 2),
        'Ret%':        _sm_ret_(eq_tks),
        'Bm Ret%':     _cpd_(ra_),
        'Contrib%':    _pc_(eq_tks),
        'Bm Contrib%': _bc_(pd.Series(nf, index=period.index), ra_),
    })

    extra_tbl   = pd.DataFrame([ts_attr],  index=['Total Stock'], columns=COLS)
    extra_stats = pd.DataFrame([ts_stats], index=['Total Stock'], columns=STAT_COLS)

    if not equity_only and 'Fixed Income' in tbl.index:
        tb_tbl         = tbl.loc[['Fixed Income']].copy()
        tb_stats       = stats.loc[['Fixed Income']].copy()
        tb_tbl.index   = ['Total Bond']
        tb_stats.index = ['Total Bond']
        extra_tbl      = pd.concat([extra_tbl,   tb_tbl])
        extra_stats    = pd.concat([extra_stats, tb_stats])

    tbl   = pd.concat([tbl,   extra_tbl])
    stats = pd.concat([stats, extra_stats])

    tbl = pd.concat([stats, tbl], axis=1)
    return tbl.round(3)


@functools.lru_cache(maxsize=4)
def compute_tier1_trades(neutral_equity=0.60, path=FILE, rebal_sheet='Rebalances'):
    """
    Month-by-month Tier I equity/bond timing trade log.

    Returns a DataFrame indexed by month-end date with columns:
        Active Eq OW%  — avg daily active equity overweight vs neutral (positive = OW equity)
        ACWI%          — ACWI compounded monthly return
        {bond_bm}%     — bond benchmark (MUB for SGA models, AGG for TE models) compounded monthly return
        Spread%        — ACWI% − {bond_bm}% (positive = equity beat bonds)
        Tier I%        — Carino-linked Tier I contribution for the month
        Result         — ✓ bet paid off  |  ✗ bet lost  |  — near-neutral
    """
    daily = _build_daily_attribution(neutral_equity, path, rebal_sheet)

    rebalances   = load_rebalances(path, rebal_sheet)
    etf_returns  = load_etf_returns(path)

    trading_dates = etf_returns.index
    daily_weights = build_daily_weights(rebalances, trading_dates)
    meta          = get_ticker_meta(rebalances)

    tickers_held = daily_weights.columns.intersection(etf_returns.columns)
    W     = daily_weights[tickers_held]
    is_eq = pd.Series({t: meta.get(t, {}).get('asset_class', '') == 'Equity'
                       for t in tickers_held})
    eq_cols = [t for t in is_eq[is_eq].index if t in W.columns]

    bond_bm = _bond_bm(rebal_sheet)
    r_acwi = etf_returns[BM_EQUITY].reindex(trading_dates)
    r_agg  = etf_returns[bond_bm].reindex(trading_dates)
    w_eq   = W[eq_cols].sum(axis=1)

    pix       = daily.index                          # only days with valid benchmark
    active_wt = (w_eq.reindex(pix) - neutral_equity) * 100
    ra        = r_acwi.reindex(pix)
    rb        = r_agg.reindex(pix)

    def compound(s):
        return ((1 + s / 100).prod() - 1) * 100

    rows = {}
    for month, g in daily.groupby(pd.Grouper(freq='ME')):
        if g.empty:
            continue
        idx    = g.index
        linked = _apply_carino_geometric(g)
        tier1  = linked['tier1'].sum()
        ow     = active_wt.reindex(idx).mean()
        acwi_r = compound(ra.reindex(idx))
        mub_r  = compound(rb.reindex(idx))
        spread = acwi_r - mub_r
        neutral_pos = abs(ow) <= 0.1
        worked = (not neutral_pos) and ((ow > 0 and spread > 0) or (ow < 0 and spread < 0))
        rows[month] = {
            'Active Eq OW%': round(ow, 1),
            'ACWI%':         round(acwi_r, 2),
            f'{bond_bm}%':   round(mub_r, 2),
            'Spread%':       round(spread, 2),
            'Tier I%':       round(tier1, 3),
            'Result':        '—' if neutral_pos else ('✓' if worked else '✗'),
        }

    df = pd.DataFrame(rows).T
    df.index = pd.DatetimeIndex(df.index)
    df.index.name = 'date'
    return df


def compute_position_attribution(start, end, neutral_equity=0.60, path=FILE,
                                 rebal_sheet='Rebalances', equity_only=False):
    """
    Per-ticker Carino-linked attribution contributions for a date range.

    Returns a dict:
        'Tier1'       → DataFrame summarising equity vs FI timing (omitted in equity_only mode)
        'FixedIncome' → DataFrame of per-FI-ticker contributions vs MUB (omitted in equity_only mode)
        '<sm_name>'   → DataFrame of per-ticker contributions vs ACWI, one key per active sub-model

    DataFrame columns: Avg Wt% | Return% | Active Return% | Contribution%
    Contributions within each sub-model sum exactly to that sub-model's Carino-linked total.
    """
    daily  = _build_daily_attribution(neutral_equity, path, rebal_sheet, equity_only)
    period = _period_slice(daily, start, end)
    if period.empty:
        raise ValueError(f"No trading days found between {start} and {end}.")

    pix = period.index

    # Carino daily scale factors (same maths as _apply_carino_geometric)
    r_port_p = period['r_port']
    r_bm_p   = period['r_bm']
    R_P = ((1 + r_port_p / 100).prod() - 1) * 100
    R_B = ((1 + r_bm_p   / 100).prod() - 1) * 100
    geo_dec = (1 + R_P / 100) / (1 + R_B / 100) - 1
    log_dec = np.log(1 + R_P / 100) - np.log(1 + R_B / 100)

    if abs(geo_dec) < 1e-10:
        scale = pd.Series(1.0, index=pix)
    else:
        K_geo = log_dec / geo_dec
        dd    = (r_port_p - r_bm_p) / 100
        ld    = np.log(1 + r_port_p / 100) - np.log(1 + r_bm_p / 100)
        kt    = np.where(dd.abs() < 1e-10, 1.0 / (1 + r_port_p / 100), ld / dd)
        scale = pd.Series(kt / K_geo, index=pix)

    rebalances  = load_rebalances(path, rebal_sheet)
    etf_returns = load_etf_returns(path)

    trading_dates = etf_returns.index
    daily_weights = build_daily_weights(rebalances, trading_dates)
    meta          = get_ticker_meta(rebalances)

    tickers_held = daily_weights.columns.intersection(etf_returns.columns)
    W = daily_weights[tickers_held]
    R = etf_returns[tickers_held].reindex(trading_dates).fillna(0)

    is_equity  = pd.Series({t: meta.get(t, {}).get('asset_class', '') == 'Equity'
                             for t in tickers_held})
    eq_tickers = is_equity[is_equity].index.tolist()
    fi_tickers = is_equity[~is_equity].index.tolist()

    bond_bm = _bond_bm(rebal_sheet)
    r_acwi = etf_returns[BM_EQUITY].reindex(trading_dates)
    r_agg  = etf_returns[bond_bm].reindex(trading_dates)

    W_p      = W.reindex(pix)
    R_p      = R.reindex(pix)
    r_acwi_p = r_acwi.reindex(pix)
    r_agg_p  = r_agg.reindex(pix)

    # In equity-only mode normalise ticker weights by total equity weight
    if equity_only:
        eq_cols_p = [t for t in eq_tickers if t in W_p.columns]
        w_eq_total_p = W_p[eq_cols_p].sum(axis=1).replace(0, np.nan)
        W_p = W_p.copy()
        for t in eq_cols_p:
            W_p[t] = W_p[t].div(w_eq_total_p)

    acwi_period = ((1 + r_acwi_p / 100).prod() - 1) * 100
    mub_period  = ((1 + r_agg_p  / 100).prod() - 1) * 100

    def make_row(ticker, bench_p):
        w_t  = W_p[ticker] if ticker in W_p.columns else pd.Series(0.0, index=pix)
        r_t  = R_p[ticker] if ticker in R_p.columns else pd.Series(0.0, index=pix)
        held = w_t > 1e-4
        if not held.any():
            return None
        contrib        = ((w_t * (r_t - bench_p)) * scale).sum()
        avg_wt         = w_t[held].mean() * 100
        per_ret        = ((1 + r_t[held]     / 100).prod() - 1) * 100
        # Compare benchmark over the same held days so Active Return% is apples-to-apples
        bench_held_ret = ((1 + bench_p[held] / 100).prod() - 1) * 100
        return {
            'Avg Wt%':        round(avg_wt, 2),
            'Return%':        round(per_ret, 2),
            'Active Return%': round(per_ret - bench_held_ret, 2),
            'Contribution%':  round(contrib, 3),
        }

    # ── Tier I summary ──
    eq_cols = [t for t in eq_tickers if t in W_p.columns]
    fi_cols = [t for t in fi_tickers if t in W_p.columns]
    w_eq_p  = W_p[eq_cols].sum(axis=1) if eq_cols else pd.Series(0.0, index=pix)
    w_fi_p  = W_p[fi_cols].sum(axis=1) if fi_cols else pd.Series(0.0, index=pix)

    tier1_df = pd.DataFrame([
        {
            'Asset':       'Equity (ACWI)',
            'Avg Wt%':     round(w_eq_p.mean() * 100, 1),
            'Neutral Wt%': round(neutral_equity * 100, 1),
            'Active OW%':  round((w_eq_p.mean() - neutral_equity) * 100, 1),
            'Period Ret%': round(acwi_period, 2),
        },
        {
            'Asset':       f'Fixed Income ({bond_bm})',
            'Avg Wt%':     round(w_fi_p.mean() * 100, 1),
            'Neutral Wt%': round((1 - neutral_equity) * 100, 1),
            'Active OW%':  round((w_fi_p.mean() - (1 - neutral_equity)) * 100, 1),
            'Period Ret%': round(mub_period, 2),
        },
    ]).set_index('Asset')

    # ── Fixed Income positions ──
    fi_rows = {t: make_row(t, r_agg_p) for t in fi_tickers}
    fi_rows = {t: v for t, v in fi_rows.items() if v is not None}
    fi_df = (pd.DataFrame(fi_rows).T.rename_axis('Ticker').sort_values('Contribution%')
             if fi_rows else pd.DataFrame())

    # ── Equity sub-model positions ──
    results = {} if equity_only else {'Tier1': tier1_df, 'FixedIncome': fi_df}
    for sm in sorted(EQUITY_SUBMODELS):
        sm_tickers = [t for t in eq_tickers
                      if meta.get(t, {}).get('sub_model', '') == sm]
        sm_rows = {t: make_row(t, r_acwi_p) for t in sm_tickers}
        sm_rows = {t: v for t, v in sm_rows.items() if v is not None}
        if sm_rows:
            results[sm] = (pd.DataFrame(sm_rows).T
                           .rename_axis('Ticker')
                           .sort_values('Contribution%'))
    return results


# ── Quick sanity check ────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=4)
def _build_tax_history(ltcg_rate=0.238, stcg_rate=0.438, neutral_equity=0.60, path=FILE,
                       accrual=False):
    """
    Replay the full portfolio history from inception, tracking FIFO cost-basis lots
    and realised CGT triggered by each rebalancing trade.

    Parameters
    ----------
    accrual : bool
        False (default) = realization basis: taxes deducted at point of sale.
        True            = accrual / mark-to-market basis: deferred-tax liability
                          (DTL = unrealised gains × applicable rate) is subtracted
                          from portfolio value daily, so tax drag accrues smoothly
                          rather than spiking at each rebalance.

    Returns
    -------
    daily     : DataFrame(r_port, r_bm, tax_paid[, v_net_accrual]) indexed by trading date.
    trade_log : DataFrame of individual lot-level realised-gain events.
    """
    rebalances   = load_rebalances(path)
    etf_returns  = load_etf_returns(path)

    trading_dates = etf_returns.index
    daily_weights = build_daily_weights(rebalances, trading_dates)
    meta          = get_ticker_meta(rebalances)

    tickers_held = daily_weights.columns.intersection(etf_returns.columns)
    W  = daily_weights[tickers_held]
    R  = etf_returns[tickers_held].reindex(trading_dates).fillna(0)

    r_acwi = etf_returns[BM_EQUITY].reindex(trading_dates)
    r_agg  = etf_returns[BM_BOND].reindex(trading_dates)
    r_bm   = neutral_equity * r_acwi + (1 - neutral_equity) * r_agg
    r_port = (W * R).sum(axis=1)

    # Cumulative price indices for each ticker (1.0 at first trading day)
    price_idx = (1 + R / 100).cumprod()

    # Pre-tax portfolio value ($1 at inception)
    port_val = (1 + r_port / 100).cumprod()

    # Rebalance dates: trading days where target weights shift materially
    w_change   = W.diff().abs().sum(axis=1)
    rebal_dates = w_change[w_change > 1e-4].index.tolist()

    # FIFO lots: {ticker: [[purchase_date, purchase_price_idx, units]]}
    # units × purchase_price ≈ $ of the $1 inception portfolio allocated to this lot
    lots          = {}
    lot_snapshots = {}  # rebal_date -> shallow copy of lots after that rebalance
    tax_paid      = pd.Series(0.0, index=trading_dates, dtype=float)
    trade_rows    = []

    prev_date    = None
    prev_weights = {}  # target weights in effect after the previous rebalance

    for rebal_date in rebal_dates:
        V     = float(port_val[rebal_date])
        W_new = W.loc[rebal_date]

        # Actual (drifted) dollar values just before this rebalance
        if prev_date is None:
            actual_dollars = {t: 0.0 for t in tickers_held}
        else:
            V_prev = float(port_val[prev_date])
            actual_dollars = {}
            for t in tickers_held:
                w = prev_weights.get(t, 0.0)
                if w < 1e-8:
                    actual_dollars[t] = 0.0
                    continue
                p_prev = float(price_idx.loc[prev_date, t])
                p_curr = float(price_idx.loc[rebal_date, t])
                actual_dollars[t] = w * V_prev * (p_curr / p_prev)

        # ── Tier I fraction per ticker ──
        # Portion of each sell driven by the aggregate equity/bond allocation shift
        # rather than within-equity sub-model rebalancing.
        equity_tks     = {t for t in tickers_held
                          if meta.get(t, {}).get('sub_model', '') in EQUITY_SUBMODELS}
        nonequity_tks  = set(tickers_held) - equity_tks

        actual_eq_tot  = sum(actual_dollars.get(t, 0.0) for t in equity_tks)
        target_eq_tot  = sum(float(W_new[t]) * V         for t in equity_tks)
        actual_neq_tot = sum(actual_dollars.get(t, 0.0) for t in nonequity_tks)
        target_neq_tot = sum(float(W_new[t]) * V         for t in nonequity_tks)

        # Pro-rata Tier I sell for each ticker (> 0 only when that side is being reduced)
        tier1_frac_map: dict = {}
        for t in tickers_held:
            act_d  = actual_dollars.get(t, 0.0)
            tgt_d  = float(W_new[t]) * V
            delta_t = tgt_d - act_d
            if delta_t >= -1e-6:          # buy or hold — no sell
                tier1_frac_map[t] = 0.0
                continue
            if t in equity_tks and actual_eq_tot > 1e-8:
                tier1_sell = max(0.0, actual_eq_tot - target_eq_tot)
                pro_rata   = act_d / actual_eq_tot * tier1_sell
            elif t in nonequity_tks and actual_neq_tot > 1e-8:
                tier1_sell = max(0.0, actual_neq_tot - target_neq_tot)
                pro_rata   = act_d / actual_neq_tot * tier1_sell
            else:
                pro_rata = 0.0
            tier1_frac_map[t] = min(1.0, pro_rata / abs(delta_t))

        for ticker in tickers_held:
            actual_d = actual_dollars.get(ticker, 0.0)
            target_d = float(W_new[ticker]) * V
            delta    = target_d - actual_d
            p_curr   = float(price_idx.loc[rebal_date, ticker])

            if delta < -1e-6:          # ── SELL ──
                units_to_sell = (-delta) / p_curr
                ticker_lots   = lots.get(ticker, [])
                remaining     = units_to_sell
                t1_frac       = tier1_frac_map.get(ticker, 0.0)

                while remaining > 1e-10 and ticker_lots:
                    lot_date, lot_price, lot_units = ticker_lots[0]
                    n       = min(lot_units, remaining)
                    gain    = n * (p_curr - lot_price)
                    holding = (rebal_date - lot_date).days

                    if gain > 1e-10:
                        rate  = ltcg_rate if holding >= 365 else stcg_rate
                        t_amt = gain * rate
                        tax_paid[rebal_date] += t_amt
                        trade_rows.append({
                            'date':         rebal_date,
                            'ticker':       ticker,
                            'sub_model':    meta.get(ticker, {}).get('sub_model', ''),
                            'lot_date':     lot_date,
                            'holding_days': holding,
                            'rate_type':    'LT' if holding >= 365 else 'ST',
                            'gain_$':       round(gain,   7),
                            'tax_$':        round(t_amt,  7),
                            'tier1_frac':   round(t1_frac, 6),
                        })

                    if n >= lot_units - 1e-10:
                        ticker_lots.pop(0)
                    else:
                        ticker_lots[0] = [lot_date, lot_price, lot_units - n]
                    remaining -= n

                lots[ticker] = ticker_lots

            elif delta > 1e-6:         # ── BUY ──
                units = delta / p_curr
                if ticker not in lots:
                    lots[ticker] = []
                lots[ticker].append([rebal_date, p_curr, units])

        prev_date    = rebal_date
        prev_weights = {t: float(W_new[t]) for t in tickers_held}

        # Capture lot state after this rebalance (for accrual DTL computation)
        if accrual:
            lot_snapshots[rebal_date] = {t: [lot[:] for lot in tl]
                                         for t, tl in lots.items()}

    _tlog_cols = ['date', 'ticker', 'sub_model', 'lot_date',
                  'holding_days', 'rate_type', 'gain_$', 'tax_$', 'tier1_frac']
    trade_log = (pd.DataFrame(trade_rows, columns=_tlog_cols)
                 if trade_rows else
                 pd.DataFrame(columns=_tlog_cols))

    daily = pd.DataFrame({'r_port': r_port, 'r_bm': r_bm, 'tax_paid': tax_paid})

    if accrual:
        # ── After-realized-tax portfolio value series ──
        # Walk every trading day: grow pre-tax, deduct cash taxes at rebalance dates.
        v_r = 1.0
        v_realized = pd.Series(0.0, index=trading_dates, dtype=float)
        for date in trading_dates:
            v_r *= (1 + float(r_port[date]) / 100)
            v_r -= float(tax_paid[date])
            v_realized[date] = v_r

        # ── Deferred tax liability (DTL) series ──
        # Lots are constant between rebalances; snapshots give us lot state at each date.
        snap_dates = sorted(lot_snapshots.keys())
        snap_idx   = 0
        current_snap: dict = {}
        dtl_series = pd.Series(0.0, index=trading_dates, dtype=float)

        for date in trading_dates:
            while snap_idx < len(snap_dates) and snap_dates[snap_idx] <= date:
                current_snap = lot_snapshots[snap_dates[snap_idx]]
                snap_idx += 1

            dtl = 0.0
            for ticker, ticker_lots in current_snap.items():
                if ticker not in price_idx.columns:
                    continue
                try:
                    p_curr = float(price_idx.loc[date, ticker])
                except KeyError:
                    continue
                for lot_date, lot_price, lot_units in ticker_lots:
                    gain = lot_units * (p_curr - lot_price)
                    if gain > 0:
                        holding = (date - lot_date).days
                        rate = ltcg_rate if holding >= 365 else stcg_rate
                        dtl += gain * rate
            dtl_series[date] = dtl

        daily['v_realized']    = v_realized
        daily['dtl']           = dtl_series
        daily['v_net_accrual'] = v_realized - dtl_series

    return daily, trade_log, port_val


@functools.lru_cache(maxsize=2)
def compute_positioning(path=FILE):
    """
    Returns (tier1_df, tier2_df) — target weights at each rebalance date (%).

    tier1_df : index=date, cols=['Equity','Fixed Income'] (+ 'Cash' if present)
    tier2_df : index=date, cols=[sub_model ...] ordered for display
    """
    rebalances    = load_rebalances(path)
    etf_returns   = load_etf_returns(path)
    trading_dates = etf_returns.index
    meta          = get_ticker_meta(rebalances)

    W = build_daily_weights(rebalances, trading_dates)
    W = W[W.columns.intersection(etf_returns.columns)]

    w_change    = W.diff().abs().sum(axis=1)
    rebal_dates = w_change[w_change > 1e-4].index
    W_r         = W.loc[rebal_dates] * 100          # convert to %

    # Normalise each row to 100% — guards against tickers present in the rebalances
    # sheet but absent from etf_returns (dropped by the intersection above), and
    # against duplicate rebalance entries that produce sums > 100%.
    row_sums = W_r.sum(axis=1).replace(0, 1)       # avoid div-by-zero
    W_r = W_r.div(row_sums, axis=0) * 100

    # Tier I — use asset_class from meta
    def asset_cls(t):
        return meta.get(t, {}).get('asset_class', '')

    equity_tks = [t for t in W_r.columns if asset_cls(t) == 'Equity']
    fi_tks     = [t for t in W_r.columns if asset_cls(t) == 'FixedIncome']
    cash_tks   = [t for t in W_r.columns if asset_cls(t) not in ('Equity', 'FixedIncome')]

    tier1 = pd.DataFrame(index=rebal_dates)
    tier1['Equity']       = W_r[equity_tks].sum(axis=1)
    tier1['Fixed Income'] = W_r[fi_tks].sum(axis=1)
    if cash_tks and W_r[cash_tks].sum().sum() > 0.1:
        tier1['Cash'] = W_r[cash_tks].sum(axis=1)

    # Tier II — aggregate by sub_model
    sm_buckets: dict = {}
    for t in W_r.columns:
        raw = meta.get(t, {}).get('sub_model', '')
        sm  = 'Fixed Income' if raw == 'FixedIncome' else (raw or 'Cash')
        sm_buckets.setdefault(sm, []).append(t)

    tier2 = pd.DataFrame(
        {sm: W_r[tks].sum(axis=1) for sm, tks in sm_buckets.items()},
        index=rebal_dates,
    )

    # Canonical column order: FI first (bottom of stack), then equity models
    SM_ORDER = ['Fixed Income', 'GV', 'OMFL', 'Sector', 'T1F', 'SGA',
                'EAFE', 'EM', 'SmallCap', 'Art', 'REIT', 'Gold', 'Cash']
    ordered = [c for c in SM_ORDER if c in tier2.columns]
    ordered += [c for c in tier2.columns if c not in ordered]
    tier2 = tier2[ordered]

    return tier1, tier2


def compute_aftertax_summary(start, end, neutral_equity=0.60,
                              ltcg_rate=0.238, stcg_rate=0.438, path=FILE,
                              accrual=False):
    """
    After-tax portfolio return for [start, end], accounting for realised CGT
    from rebalancing trades (dividends excluded).

    Parameters
    ----------
    accrual : bool
        False = realization basis (taxes hit at point of sale).
        True  = accrual / mark-to-market (DTL accrues daily as unrealised gains build).

    Returns a dict with the same keys in both modes.
    """
    daily, trade_log, port_val_full = _build_tax_history(
        ltcg_rate, stcg_rate, neutral_equity, path, accrual)

    period = _period_slice(daily, start, end)
    if period.empty:
        raise ValueError(f"No trading days between {start} and {end}.")

    def compound(s):
        return ((1 + s / 100).prod() - 1) * 100

    pretax_ret = compound(period['r_port'])
    bm_ret     = compound(period['r_bm'])

    if accrual:
        v_net = daily['v_net_accrual']
        v0    = float(v_net[period.index[0]])
        v1    = float(v_net[period.index[-1]])
        aftertax_ret = (v1 / v0 - 1) * 100
    else:
        # Scale: taxes are stored in inception-$1 units; convert to period-start-$1 units
        V_start = float(port_val_full[period.index[0]])

        # Walk the period day-by-day: grow pre-tax each day, deduct taxes at rebalance dates
        atv = 1.0
        for date in period.index:
            atv *= (1 + float(period.loc[date, 'r_port']) / 100)
            tax  = float(period.loc[date, 'tax_paid'])
            if tax > 0:
                atv -= tax / V_start
        aftertax_ret = (atv - 1) * 100

    tax_drag = aftertax_ret - pretax_ret

    # Trade log and realized tax breakdown (same for both modes —
    # accrual still shows the actual trades that occurred)
    V_start_real = float(port_val_full[period.index[0]])

    # Build period trade log with taxes expressed as % of period-start portfolio
    mask = ((trade_log['date'] > pd.Timestamp(start)) &
            (trade_log['date'] <= pd.Timestamp(end)))
    plog = trade_log[mask].copy()

    if not plog.empty:
        plog['gain_%'] = (plog['gain_$'] / V_start_real * 100).round(4)
        plog['tax_%']  = (plog['tax_$']  / V_start_real * 100).round(4)
        plog = plog.drop(columns=['gain_$', 'tax_$'])
        lt_tax = float(plog.loc[plog['rate_type'] == 'LT', 'tax_%'].sum())
        st_tax = float(plog.loc[plog['rate_type'] == 'ST', 'tax_%'].sum())
    else:
        lt_tax = st_tax = 0.0

    return {
        'pretax_return':    round(pretax_ret,    3),
        'aftertax_return':  round(aftertax_ret,  3),
        'benchmark_return': round(bm_ret,         3),
        'tax_drag':         round(tax_drag,       3),
        'total_tax_pct':    round(lt_tax + st_tax, 3),
        'lt_tax_pct':       round(lt_tax,          3),
        'st_tax_pct':       round(st_tax,          3),
        'trade_log':        plog,
    }


if __name__ == '__main__':
    print("Loading data...")
    reb  = load_rebalances()
    etf  = load_etf_returns()
    acwi = load_acwi_weights()

    print(f"  Rebalances: {len(reb)} rows, {reb['date'].nunique()} rebalance events")
    print(f"  ETF returns: {etf.shape[0]} trading days, {etf.shape[1]} tickers")
    print(f"  ACWI weights: {acwi.shape[0]} months")

    print("\nSub-models found in rebalances:")
    print(sorted(reb['sub_model'].unique()))

    print("\nComputing monthly attribution (Balanced, 60/40)...")
    attr = compute_monthly_attribution(neutral_equity=0.60)
    print(f"  {len(attr)} months ({attr.index[0].date()} to {attr.index[-1].date()})")

    primary_cols = ['portfolio_return', 'benchmark_return', 'excess_return',
                    'tier1', 'fi_selection'] + sorted(EQUITY_SUBMODELS & set(attr.columns))
    print("\nAnnualised mean attribution (×12):")
    print((attr[primary_cols].mean() * 12).round(3).to_string())

    print("\nTwo-level attribution table: 2025-06-01 to 2026-05-29 (Balanced 60/40)...")
    pd.set_option('display.float_format', '{:8.3f}'.format)
    pd.set_option('display.max_colwidth', 50)
    tbl = compute_attribution_table('2025-06-01', '2026-05-29', neutral_equity=0.60)
    print(tbl.to_string())
