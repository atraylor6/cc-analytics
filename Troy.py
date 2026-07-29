import pandas as pd
import numpy as np
import fredapi
import matplotlib.pyplot as plt
from itertools import product
from fredapi import Fred

# Define constants
ROLLING_WINDOW_1YR = 12
API_KEY = 'd0fc5bc2297df338f8f31e08b197b1d9'
BENCHMARK_COLUMN = 'benchmark'
TE_MAX = 10.0

# Define helper functions
def annualized_return(returns):
    product_returns = (1 + returns).prod()
    total_rows = returns.shape[0]
    geometric_mean_return = (product_returns ** (12 / total_rows)) - 1
    return geometric_mean_return * 100

def rolling_annualized_return(returns, window_size):
    log_growth = np.log1p(returns).rolling(window=window_size).sum()
    return (np.exp(log_growth * (12 / window_size)) - 1) * 100

def calculate_rolling_excess_returns(data, window_size, benchmark_returns):
    rolling_excess_returns = rolling_annualized_return(data, window_size).sub(
        rolling_annualized_return(benchmark_returns, window_size), axis=0
    )
    return rolling_excess_returns.dropna()

def annualized_standard_deviation(returns):
    monthly_std = returns.std()
    annual_std = (monthly_std * (12 ** 0.5)) * 100
    return annual_std

def downside_deviation(excess_returns):
    downside_squared = (excess_returns.where(excess_returns < 0) ** 2).sum()
    num_values = excess_returns.notna().sum()
    dev = downside_squared / num_values
    sqrt_dev = np.sqrt(dev)
    annualized_sqrt_dev = sqrt_dev * np.sqrt(12)
    return annualized_sqrt_dev * 100

def tracking_error(excess_returns):
    tracking_error_value = (excess_returns.std() * np.sqrt(12)) * 100
    return tracking_error_value

def average_10yr(df, api_key):
    fred = Fred(api_key=api_key)
    start_date = df.index.min().strftime('%Y-%m-%d')
    end_date = df.index.max().strftime('%Y-%m-%d')
    series_id = 'GS10'
    data = fred.get_series(series_id, start_date, end_date)
    average_rate = data.mean()
    return average_rate

def sharpe_ratio(returns, average_10yr_rate):
    annualized_returns = returns.apply(annualized_return)
    excess_return = annualized_returns - average_10yr_rate
    std_dev = annualized_standard_deviation(returns)
    sharpe_ratio_value = excess_return / std_dev
    return sharpe_ratio_value

def mean_excess_return(portfolio_returns, benchmark_returns):
    portfolio_annualized = annualized_return(portfolio_returns)
    benchmark_annualized = annualized_return(benchmark_returns)
    excess_return = portfolio_annualized - benchmark_annualized
    return excess_return

def information_ratio(returns, benchmark_returns):
    mean_excess_return_value = mean_excess_return(returns, benchmark_returns)
    excess_returns = returns.subtract(benchmark_returns, axis=0)
    tracking_errors = tracking_error(excess_returns)
    information_ratio_value = mean_excess_return_value / tracking_errors
    return information_ratio_value

def sortino_ratio(returns, benchmark_returns):
    excess_return = mean_excess_return(returns, benchmark_returns)
    downside_deviation_value = downside_deviation(returns - benchmark_returns)
    sortino_ratio_value = excess_return / downside_deviation_value
    return sortino_ratio_value

def hit_rate(rolling_excess_returns, threshold=1):
    hit_rate_value = (rolling_excess_returns > threshold).sum() / rolling_excess_returns.count()
    return hit_rate_value * 100

def ic(hit_rate_value):
    ic_value = 2 * (hit_rate_value / 100) - 1
    return ic_value

def confidence_ir(ic_value, information_ratio_value):
    confidence_ir_value = ic_value * information_ratio_value
    return confidence_ir_value

def confidence_sharpe_ratio(ic_value, sharpe_ratio_value):
    confidence_sharpe_ratio_value = ic_value * sharpe_ratio_value
    return confidence_sharpe_ratio_value

def confidence_sortino_ratio(ic_value, sortino_ratio_value):
    confidence_sortino_ratio_value = ic_value * sortino_ratio_value
    return confidence_sortino_ratio_value

def run_scenario(min_allocations, max_allocations, label, returns, benchmark_returns, asset_names):
    weight_combinations = [
        p for p in product(range(0, 105, 5), repeat=len(asset_names))
        if sum(p) == 100 and all(min_allocations[asset_names[i]] <= p[i] <= max_allocations[asset_names[i]] for i in range(len(asset_names)))
    ]
    if not weight_combinations:
        raise ValueError(f"No weight combinations satisfy the '{label}' constraints.")

    portfolio_data = {}
    for i, weights in enumerate(weight_combinations):
        weight_array = np.array(weights) / 100
        portfolio_data[f'Portfolio_{i+1}'] = returns.dot(weight_array)
    portfolio_returns = pd.DataFrame(portfolio_data, index=returns.index)

    rolling_excess_returns_1yr = calculate_rolling_excess_returns(portfolio_returns, ROLLING_WINDOW_1YR, benchmark_returns)
    hit_rate_1yr = hit_rate(rolling_excess_returns_1yr)
    ic_1yr = ic(hit_rate_1yr)

    mean_excess_returns = mean_excess_return(portfolio_returns, benchmark_returns)

    excess_returns = portfolio_returns.subtract(benchmark_returns, axis=0)
    tracking_errors = tracking_error(excess_returns)

    information_ratios = information_ratio(portfolio_returns, benchmark_returns)
    confidence_ir_1yr = confidence_ir(ic_1yr, information_ratios)

    downside_deviations = downside_deviation(excess_returns)

    allocations_df = pd.DataFrame(weight_combinations, columns=returns.columns, index=portfolio_returns.columns)
    results_df = pd.concat([confidence_ir_1yr, downside_deviations, ic_1yr, mean_excess_returns, tracking_errors, information_ratios, allocations_df], axis=1)
    results_df.columns = ['Confidence IR', 'Downside Deviation', 'IC', 'Mean Excess Return', 'Tracking Error', 'Information Ratio'] + list(returns.columns)

    # Pareto-efficient frontier: lowest achievable TE up to TE_MAX
    sorted_by_te = results_df.sort_values('Tracking Error').reset_index(drop=True)
    prev_max = sorted_by_te['Mean Excess Return'].cummax().shift(1)
    is_pareto = sorted_by_te['Mean Excess Return'] > prev_max.fillna(-np.inf)
    pareto_frontier = sorted_by_te[is_pareto]

    te_min = results_df['Tracking Error'].min()
    te_targets = [te_min] + [t for t in range(int(np.ceil(te_min)), int(TE_MAX) + 1) if t > te_min]

    frontier_table_rows = []
    for te_target in te_targets:
        eligible = results_df[results_df['Tracking Error'] <= te_target]
        if eligible.empty:
            print(f"[{label}] No portfolio achieves Tracking Error <= {te_target:.2f}%; skipping.")
            continue
        best = eligible.loc[eligible['Mean Excess Return'].idxmax()]
        row = {
            'Target TE (%)': round(te_target, 2),
            'Tracking Error': best['Tracking Error'],
            'Excess Return': best['Mean Excess Return'],
            'Information Ratio': best['Information Ratio'],
        }
        for asset in asset_names:
            row[asset] = best[asset]
        frontier_table_rows.append(row)
    frontier_table = pd.DataFrame(frontier_table_rows).set_index('Target TE (%)')

    sorted_results = results_df.sort_values(by='Confidence IR', ascending=False)
    top_10_portfolios = sorted_results.head(10)
    top_3_portfolios = top_10_portfolios.sort_values(by='Downside Deviation', ascending=True).head(3)
    optimal_portfolio = top_3_portfolios.sort_values(by='IC', ascending=False).head(1)

    return {
        'label': label,
        'results_df': results_df,
        'pareto_frontier': pareto_frontier,
        'te_min': te_min,
        'frontier_table': frontier_table,
        'top_10_portfolios': top_10_portfolios,
        'optimal_portfolio': optimal_portfolio,
    }

def main():
    # Load data
    df = pd.read_excel("troydata.xlsx", sheet_name="Sheet2")
    df.set_index("date", inplace=True)

    returns = df.drop(columns=[BENCHMARK_COLUMN])
    benchmark_returns = df[BENCHMARK_COLUMN]
    asset_names = returns.columns.tolist()

    average_10yr_rate = average_10yr(df, API_KEY)

    # Unconstrained scenario: 0%-100% on every asset
    unconstrained_min = {asset: 0.0 for asset in asset_names}
    unconstrained_max = {asset: 100.0 for asset in asset_names}

    # Constrained scenario: allocation limits from the user
    constrained_min = {}
    constrained_max = {}
    print("Specify the minimum and maximum allocation for each asset (in %) for the constrained scenario:")
    for asset in asset_names:
        constrained_min[asset] = float(input(f"Enter minimum allocation for {asset}: "))
        constrained_max[asset] = float(input(f"Enter maximum allocation for {asset}: "))

    scenarios = [
        run_scenario(unconstrained_min, unconstrained_max, 'Unconstrained', returns, benchmark_returns, asset_names),
        run_scenario(constrained_min, constrained_max, 'Constrained', returns, benchmark_returns, asset_names),
    ]

    SCENARIO_COLORS = {'Unconstrained': '#2a78d6', 'Constrained': '#1baf7a'}

    fig, ax = plt.subplots(figsize=(10, 6))
    overall_max_te = TE_MAX

    for scenario in scenarios:
        label = scenario['label']
        color = SCENARIO_COLORS[label]
        pareto_frontier = scenario['pareto_frontier']
        te_min = scenario['te_min']
        frontier_table = scenario['frontier_table']
        overall_max_te = max(overall_max_te, scenario['results_df']['Tracking Error'].max())

        te_grid = np.arange(te_min, TE_MAX + 0.05, 0.1)
        pareto_te = pareto_frontier['Tracking Error'].to_numpy()
        pareto_xr = pareto_frontier['Mean Excess Return'].to_numpy()
        grid_idx = np.searchsorted(pareto_te, te_grid, side='right') - 1
        frontier_curve = np.where(grid_idx >= 0, pareto_xr[np.clip(grid_idx, 0, len(pareto_xr) - 1)], np.nan)

        ax.plot(te_grid, frontier_curve, color=color, linewidth=2.5, label=label)
        ax.scatter(frontier_table['Tracking Error'], frontier_table['Excess Return'], color=color, s=30, zorder=5)

        label_offset = (6, 6) if label == 'Unconstrained' else (6, -14)
        plotted_points = frontier_table.drop_duplicates(subset=['Tracking Error', 'Excess Return'])
        for te_target, row in plotted_points.iterrows():
            ax.annotate(
                f"{row['Excess Return']:.2f}% | IR {row['Information Ratio']:.2f}",
                (row['Tracking Error'], row['Excess Return']),
                textcoords='offset points', xytext=label_offset, fontsize=7.5, color='#0b0b0b',
            )

        print(f"\n[{label}] Tracking-error efficient frontier ({te_min:.2f}% [lowest achievable] to {TE_MAX:.0f}%):\n")
        print(frontier_table.to_string())

    ax.set_xlim(0, overall_max_te * 1.05)
    ax.set_xlabel('Tracking Error (%)', color='#52514e')
    ax.set_ylabel('Excess Return vs Benchmark (%)', color='#52514e')
    ax.set_title('Efficient Frontier: Unconstrained vs. Constrained', color='#0b0b0b')
    ax.grid(True, alpha=0.3, color='#e1e0d9')
    ax.legend(loc='best', frameon=False)
    fig.tight_layout()
    fig.savefig('troy_frontier.png', dpi=150)
    plt.show()

    for scenario in scenarios:
        label = scenario['label']
        top_10_portfolios = scenario['top_10_portfolios']
        optimal_portfolio = scenario['optimal_portfolio']
        final_output = optimal_portfolio[['Confidence IR', 'Downside Deviation', 'IC', 'Mean Excess Return'] + list(returns.columns)]

        print(f"\n[{label}] Optimal Portfolio:")
        print(final_output)
        print(top_10_portfolios)

    # --- Export results to Excel ---
    with pd.ExcelWriter('troy_output.xlsx', engine='openpyxl') as writer:
        for scenario in scenarios:
            label = scenario['label']
            scenario['frontier_table'].to_excel(writer, sheet_name=f'{label} Frontier')
            scenario['top_10_portfolios'].rename_axis('Portfolio').to_excel(writer, sheet_name=f'{label} Top 10')
            scenario['optimal_portfolio'].rename_axis('Portfolio').to_excel(writer, sheet_name=f'{label} Optimal')

    print("\nSaved results to troy_output.xlsx")


if __name__ == "__main__":
    main()
