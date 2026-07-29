"""Tests for Troy.py's portfolio-optimization calculations.

Two layers of validation:
  1. Direct unit tests on Troy's pure functions with synthetic data
     (same style/cases as tests/test_cc_data.py).
  2. Cross-validation against cc_data.py's already-tested statistics
     functions, which serve as an independent reference implementation.

Unit convention: Troy's functions take DECIMAL returns (0.01 = 1%), while
cc_data's functions take PERCENT returns (1.0 = 1%). Cross-checks multiply
Troy's inputs by 100 before handing them to cc_data so both sides describe
the same underlying returns.
"""

import numpy as np
import pandas as pd
import pytest

import cc_data
from Troy import (
    annualized_return,
    calculate_rolling_excess_returns,
    confidence_ir,
    downside_deviation,
    hit_rate,
    ic,
    information_ratio,
    mean_excess_return,
    rolling_annualized_return,
    run_scenario,
    tracking_error,
)


# ── annualized_return ─────────────────────────────────────────────────────────

def test_annualized_return_geometric_compounding():
    """12 months of 1% (decimal 0.01) should compound to (1.01^12 - 1)*100."""
    returns = pd.Series([0.01] * 12)
    expected = ((1.01 ** 12) - 1) * 100
    assert annualized_return(returns) == pytest.approx(expected, rel=1e-4)


def test_annualized_return_zero_returns():
    assert annualized_return(pd.Series([0.0] * 12)) == pytest.approx(0.0)


def test_annualized_return_positive_for_net_positive_series():
    returns = pd.Series([0.01, -0.005, 0.02, -0.003, 0.015] * 2)
    assert annualized_return(returns) > 0


def test_annualized_return_negative_for_net_negative_series():
    returns = pd.Series([-0.01, 0.005, -0.02, 0.003, -0.015] * 2)
    assert annualized_return(returns) < 0


def test_annualized_return_vectorizes_across_dataframe_columns():
    df = pd.DataFrame({"A": [0.01] * 12, "B": [0.02] * 12})
    result = annualized_return(df)
    assert result["A"] == pytest.approx(((1.01 ** 12) - 1) * 100, rel=1e-4)
    assert result["B"] == pytest.approx(((1.02 ** 12) - 1) * 100, rel=1e-4)


def test_annualized_return_matches_cc_data_reference():
    returns = pd.Series([0.012, -0.008, 0.021, 0.004, -0.015, 0.009] * 4)
    troy_value = annualized_return(returns)
    cc_value = cc_data.annualized_return(returns * 100)
    assert troy_value == pytest.approx(cc_value, rel=1e-9)


# ── tracking_error ────────────────────────────────────────────────────────────

def test_tracking_error_identical_series_is_zero():
    excess = pd.Series([0.0, 0.0, 0.0, 0.0])
    assert tracking_error(excess) == pytest.approx(0.0)


def test_tracking_error_positive_for_different_series():
    portfolio = pd.Series([0.01, -0.01, 0.02, -0.02])
    benchmark = pd.Series([0.005, -0.005, 0.015, -0.015])
    assert tracking_error(portfolio - benchmark) > 0


def test_tracking_error_matches_cc_data_reference():
    portfolio = pd.Series([0.012, -0.008, 0.021, 0.004, -0.015, 0.009] * 4)
    benchmark = pd.Series([0.010, -0.006, 0.018, 0.002, -0.012, 0.007] * 4)
    troy_value = tracking_error(portfolio - benchmark)
    cc_value = cc_data.tracking_error(portfolio * 100, benchmark * 100)
    assert troy_value == pytest.approx(cc_value, rel=1e-9)


# ── downside_deviation ────────────────────────────────────────────────────────

def test_downside_deviation_all_non_negative_is_zero():
    excess = pd.Series([0.01, 0.02, 0.03, 0.005])
    assert downside_deviation(excess) == pytest.approx(0.0)


def test_downside_deviation_positive_when_negative_excess_present():
    excess = pd.Series([0.01, -0.02, 0.03, -0.01])
    assert downside_deviation(excess) > 0


def test_downside_deviation_matches_cc_data_reference():
    excess = pd.Series([0.012, -0.008, 0.021, 0.004, -0.015, 0.009] * 4)
    troy_value = downside_deviation(excess)
    cc_value = cc_data.downside_deviation(excess * 100, mar=0)
    assert troy_value == pytest.approx(cc_value, rel=1e-9)


def test_downside_deviation_vectorizes_across_dataframe_columns():
    df = pd.DataFrame({
        "A": [0.01, -0.02, 0.03, -0.01],
        "B": [0.01, 0.02, 0.03, 0.005],
    })
    result = downside_deviation(df)
    assert result["A"] > 0
    assert result["B"] == pytest.approx(0.0)


# ── mean_excess_return ────────────────────────────────────────────────────────

def test_mean_excess_return_outperforming_is_positive():
    portfolio = pd.Series([0.02] * 12)
    benchmark = pd.Series([0.01] * 12)
    assert mean_excess_return(portfolio, benchmark) > 0


def test_mean_excess_return_identical_series_is_zero():
    r = pd.Series([0.01, -0.005, 0.02] * 4)
    assert mean_excess_return(r, r.copy()) == pytest.approx(0.0)


def test_mean_excess_return_matches_cc_data_reference():
    portfolio = pd.Series([0.012, -0.008, 0.021, 0.004, -0.015, 0.009] * 4)
    benchmark = pd.Series([0.010, -0.006, 0.018, 0.002, -0.012, 0.007] * 4)
    troy_value = mean_excess_return(portfolio, benchmark)
    cc_value = cc_data.excess_return(portfolio * 100, benchmark * 100)
    assert troy_value == pytest.approx(cc_value, rel=1e-9)


# ── information_ratio ─────────────────────────────────────────────────────────

def test_information_ratio_matches_cc_data_reference():
    portfolio = pd.Series([0.012, -0.008, 0.021, 0.004, -0.015, 0.009] * 4)
    benchmark = pd.Series([0.010, -0.006, 0.018, 0.002, -0.012, 0.007] * 4)
    troy_value = information_ratio(portfolio, benchmark)
    cc_value = cc_data.information_ratio(portfolio * 100, benchmark * 100)
    assert troy_value == pytest.approx(cc_value, rel=1e-9)


def test_information_ratio_vectorizes_across_dataframe_columns():
    benchmark = pd.Series([0.010, -0.006, 0.018, 0.002, -0.012, 0.007] * 4)
    portfolios = pd.DataFrame({
        "A": [0.012, -0.008, 0.021, 0.004, -0.015, 0.009] * 4,
        "B": [0.005, -0.003, 0.010, 0.001, -0.007, 0.004] * 4,
    })
    result = information_ratio(portfolios, benchmark)
    assert result["A"] == pytest.approx(
        cc_data.information_ratio(portfolios["A"] * 100, benchmark * 100), rel=1e-9
    )
    assert result["B"] == pytest.approx(
        cc_data.information_ratio(portfolios["B"] * 100, benchmark * 100), rel=1e-9
    )


# ── rolling_annualized_return / calculate_rolling_excess_returns ─────────────

def test_rolling_annualized_return_matches_naive_loop():
    """Locks in the vectorized (log-return) rolling calc against a direct,
    unvectorized reimplementation using annualized_return in a Python loop —
    the same equivalence verified manually before this was adopted in Troy.py."""
    np.random.seed(0)
    returns = pd.Series(np.random.normal(0.005, 0.03, 60))
    window = 12

    naive = returns.rolling(window).apply(lambda r: annualized_return(pd.Series(r)), raw=False)
    fast = rolling_annualized_return(returns, window)

    pd.testing.assert_series_equal(fast.dropna(), naive.dropna(), check_names=False, rtol=1e-8)


def test_calculate_rolling_excess_returns_zero_when_portfolio_equals_benchmark():
    returns = pd.Series(np.linspace(-0.02, 0.03, 36))
    result = calculate_rolling_excess_returns(returns, 12, returns.copy())
    assert (result.abs() < 1e-9).all()


# ── hit_rate / ic / confidence_ir ─────────────────────────────────────────────

def test_hit_rate_all_above_threshold_is_100():
    rolling_excess = pd.Series([5.0, 10.0, 3.0, 8.0])
    assert hit_rate(rolling_excess, threshold=1) == pytest.approx(100.0)


def test_hit_rate_none_above_threshold_is_zero():
    rolling_excess = pd.Series([-5.0, -10.0, -3.0, 0.5])
    assert hit_rate(rolling_excess, threshold=1) == pytest.approx(0.0)


def test_ic_range_matches_hit_rate_formula():
    # hit_rate=100 -> ic=1 ; hit_rate=0 -> ic=-1 ; hit_rate=50 -> ic=0
    assert ic(100.0) == pytest.approx(1.0)
    assert ic(0.0) == pytest.approx(-1.0)
    assert ic(50.0) == pytest.approx(0.0)


def test_confidence_ir_is_product_of_ic_and_information_ratio():
    assert confidence_ir(0.5, 2.0) == pytest.approx(1.0)
    assert confidence_ir(-0.5, 2.0) == pytest.approx(-1.0)


# ── run_scenario: structural / integration checks ─────────────────────────────

@pytest.fixture
def toy_data():
    """A small, deterministic 3-asset dataset — not real market data,
    just enough to exercise the full run_scenario pipeline quickly."""
    np.random.seed(42)
    n_months = 36
    dates = pd.date_range("2020-01-31", periods=n_months, freq="ME")
    asset_names = ["Alpha", "Beta", "Gamma"]
    returns = pd.DataFrame(
        np.random.normal(0.005, 0.02, size=(n_months, len(asset_names))),
        index=dates, columns=asset_names,
    )
    benchmark_returns = pd.Series(np.random.normal(0.006, 0.018, size=n_months), index=dates)
    return returns, benchmark_returns, asset_names


def test_run_scenario_confidence_ir_equals_ic_times_information_ratio(toy_data):
    returns, benchmark_returns, asset_names = toy_data
    min_a = {a: 0.0 for a in asset_names}
    max_a = {a: 100.0 for a in asset_names}
    result = run_scenario(min_a, max_a, "Toy", returns, benchmark_returns, asset_names)

    results_df = result["results_df"]
    recomputed = results_df["IC"] * results_df["Information Ratio"]
    pd.testing.assert_series_equal(
        results_df["Confidence IR"], recomputed, check_names=False, rtol=1e-8
    )


def test_run_scenario_frontier_is_pareto_optimal(toy_data):
    """No portfolio in the full universe should both match-or-beat a frontier
    row's Tracking Error and strictly beat its Excess Return."""
    returns, benchmark_returns, asset_names = toy_data
    min_a = {a: 0.0 for a in asset_names}
    max_a = {a: 100.0 for a in asset_names}
    result = run_scenario(min_a, max_a, "Toy", returns, benchmark_returns, asset_names)

    results_df = result["results_df"]
    for _, row in result["frontier_table"].iterrows():
        dominators = results_df[
            (results_df["Tracking Error"] <= row["Tracking Error"] + 1e-9)
            & (results_df["Mean Excess Return"] > row["Excess Return"] + 1e-9)
        ]
        assert dominators.empty


def test_run_scenario_respects_allocation_constraints(toy_data):
    returns, benchmark_returns, asset_names = toy_data
    min_a = {"Alpha": 0.0, "Beta": 0.0, "Gamma": 0.0}
    max_a = {"Alpha": 20.0, "Beta": 100.0, "Gamma": 100.0}
    result = run_scenario(min_a, max_a, "Toy", returns, benchmark_returns, asset_names)

    assert (result["results_df"]["Alpha"] <= 20.0).all()


def test_run_scenario_raises_when_no_weights_satisfy_constraints(toy_data):
    returns, benchmark_returns, asset_names = toy_data
    min_a = {"Alpha": 60.0, "Beta": 60.0, "Gamma": 60.0}  # sums > 100, impossible
    max_a = {"Alpha": 100.0, "Beta": 100.0, "Gamma": 100.0}
    with pytest.raises(ValueError):
        run_scenario(min_a, max_a, "Impossible", returns, benchmark_returns, asset_names)


# ── Cross-validation against the real troydata.xlsx dataset ──────────────────

@pytest.fixture(scope="module")
def real_troy_data():
    df = pd.read_excel("troydata.xlsx", sheet_name="Sheet2")
    df.set_index("date", inplace=True)
    return df


def test_real_data_annualized_return_matches_cc_data(real_troy_data):
    for col in ["R1", "EAFE", "EM", "MTUM", "OMFL", "Quality", "benchmark"]:
        troy_value = annualized_return(real_troy_data[col])
        cc_value = cc_data.annualized_return(real_troy_data[col] * 100)
        assert troy_value == pytest.approx(cc_value, rel=1e-9), f"mismatch for {col}"


def test_real_data_tracking_error_matches_cc_data(real_troy_data):
    benchmark = real_troy_data["benchmark"]
    for col in ["R1", "EAFE", "EM", "MTUM", "OMFL", "Quality"]:
        portfolio = real_troy_data[col]
        troy_value = tracking_error(portfolio - benchmark)
        cc_value = cc_data.tracking_error(portfolio * 100, benchmark * 100)
        assert troy_value == pytest.approx(cc_value, rel=1e-9), f"mismatch for {col}"


def test_real_data_information_ratio_matches_cc_data(real_troy_data):
    benchmark = real_troy_data["benchmark"]
    for col in ["R1", "EAFE", "EM", "MTUM", "OMFL", "Quality"]:
        portfolio = real_troy_data[col]
        troy_value = information_ratio(portfolio, benchmark)
        cc_value = cc_data.information_ratio(portfolio * 100, benchmark * 100)
        assert troy_value == pytest.approx(cc_value, rel=1e-9), f"mismatch for {col}"


def test_real_data_downside_deviation_matches_cc_data(real_troy_data):
    benchmark = real_troy_data["benchmark"]
    for col in ["R1", "EAFE", "EM", "MTUM", "OMFL", "Quality"]:
        excess = real_troy_data[col] - benchmark
        troy_value = downside_deviation(excess)
        cc_value = cc_data.downside_deviation(excess * 100, mar=0)
        assert troy_value == pytest.approx(cc_value, rel=1e-9), f"mismatch for {col}"
