import pandas as pd
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from datetime import datetime

# Read the CSV file and convert the dates to datetime objects
data = pd.read_csv("historical_data.csv")
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)
def analyze_strategies(start_date, end_date):
    # Read historical prices from the CSV file
    data = pd.read_csv("historical_data.csv")
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    # Convert start_date and end_date to datetime objects if they are provided as strings
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

    # Filter the data based on the specified start and end dates
    filtered_data = data.loc[start_date:end_date]
    return filtered_data

filtered_data = analyze_strategies('2021-01-03','2022-11-14')
    # Calculate the daily returns
returns = filtered_data.pct_change().dropna()
# Calculate the returns
returns = data.pct_change().dropna()

# Define the strategies
def risk_parity(returns, rebalance_freq):
    def calculate_portfolio_risk(weights, cov_matrix):
        return np.sqrt(weights.T @ cov_matrix @ weights)

    def risk_objective_function(weights, cov_matrix):
        return calculate_portfolio_risk(weights, cov_matrix)

    def constraint_sum_to_one(weights):
        return np.sum(weights) - 1

    rebalanced_returns = []
    prev_weights = None

    for period, period_returns in returns.resample(rebalance_freq):
        cov_matrix = period_returns.cov().to_numpy()

        if prev_weights is None:
            init_weights = np.full(cov_matrix.shape[0], 1 / cov_matrix.shape[0])
        else:
            init_weights = prev_weights

        constraints = [{'type': 'eq', 'fun': constraint_sum_to_one}]
        bounds = [(0, 1) for _ in range(cov_matrix.shape[0])]
        result = minimize(risk_objective_function, init_weights, args=(cov_matrix,),
                          bounds=bounds, constraints=constraints, method='SLSQP')

        weights = result.x
        rebalanced_period_returns = period_returns @ weights
        rebalanced_returns.append(rebalanced_period_returns)
        prev_weights = weights

    return pd.concat(rebalanced_returns)


def buy_and_hold_mvo(returns, rebalance_freq):
    def calculate_portfolio_return(weights, mean_returns):
        return np.sum(weights * mean_returns)

    def mvo_objective_function(weights, mean_returns, cov_matrix, risk_aversion):
        return risk_aversion * (weights.T @ cov_matrix @ weights) - calculate_portfolio_return(weights, mean_returns)

    def constraint_sum_to_one(weights):
        return np.sum(weights) - 1

    rebalanced_returns = []
    prev_weights = None

    for period, period_returns in returns.resample(rebalance_freq):
        mean_returns = period_returns.mean().to_numpy()
        cov_matrix = period_returns.cov().to_numpy()

        if prev_weights is None:
            init_weights = np.full(cov_matrix.shape[0], 1 / cov_matrix.shape[0])
        else:
            init_weights = prev_weights

        risk_aversion = 1
        constraints = [{'type': 'eq', 'fun': constraint_sum_to_one}]
        bounds = [(0, 1) for _ in range(cov_matrix.shape[0])]
        result = minimize(mvo_objective_function, init_weights, args=(mean_returns, cov_matrix, risk_aversion),
                          bounds=bounds, constraints=constraints, method='SLSQP')

        weights = result.x
        rebalanced_period_returns = period_returns @ weights
        rebalanced_returns.append(rebalanced_period_returns)
        prev_weights = weights

    return pd.concat(rebalanced_returns)

# Define the performance metrics
def annual_return(portfolio_returns):
    return np.mean(portfolio_returns) * 252

def annual_volatility(portfolio_returns):
    return np.std(portfolio_returns) * np.sqrt(252)

def sharpe_ratio(portfolio_returns, risk_free_rate=0.06):
    return (annual_return(portfolio_returns) - risk_free_rate) / annual_volatility(portfolio_returns)

def sortino_ratio(portfolio_returns, annual_risk_free_rate=0.06, target_return=0):
    daily_risk_free_rate = np.power(1 + annual_risk_free_rate, 1 / 252) - 1
    excess_returns = portfolio_returns - daily_risk_free_rate
    downside_returns = np.copy(excess_returns)
    downside_returns[downside_returns > target_return] = 0
    downside_risk = np.sqrt(np.mean(np.square(downside_returns)))
    daily_sortino_ratio = np.mean(excess_returns) / downside_risk
    annual_sortino_ratio = daily_sortino_ratio * np.sqrt(252) # Assuming 252 trading days in a year
    return annual_sortino_ratio

def value_at_risk(portfolio_returns, confidence_level=0.95):
    return np.percentile(portfolio_returns, 100 - confidence_level * 100)

def cvar(portfolio_returns, confidence_level=0.95):
    var = value_at_risk(portfolio_returns, confidence_level)
    return np.mean(portfolio_returns[portfolio_returns <= var])

def max_drawdown(portfolio_returns):
    cumulative_returns = (1 + portfolio_returns).cumprod()
    max_return = cumulative_returns.expanding(min_periods=1).max()
    drawdown = cumulative_returns / max_return - 1
    return drawdown.min()

# Set rebalance frequencies
risk_parity_rebalance_freq = 'M'  # Quarterly rebalancing
buy_and_hold_rebalance_freq = 'A'  # Annual rebalancing

# Implement the strategies
risk_parity_returns = risk_parity(returns, risk_parity_rebalance_freq)
buy_and_hold_returns = buy_and_hold_mvo(returns, buy_and_hold_rebalance_freq)

# Calculate the performance metrics
risk_parity_metrics = {
    'Annual Return': annual_return(risk_parity_returns),
    'Annual Volatility': annual_volatility(risk_parity_returns),
    'Sharpe Ratio': sharpe_ratio(risk_parity_returns),
    'Sortino Ratio': sortino_ratio(risk_parity_returns),
    'VaR': value_at_risk(risk_parity_returns),
    'CVaR': cvar(risk_parity_returns),
    'Max Drawdown': max_drawdown(risk_parity_returns)
}

buy_and_hold_metrics = {
    'Annual Return': annual_return(buy_and_hold_returns),
    'Annual Volatility': annual_volatility(buy_and_hold_returns),
    'Sharpe Ratio': sharpe_ratio(buy_and_hold_returns),
    'Sortino Ratio': sortino_ratio(buy_and_hold_returns),
    'VaR': value_at_risk(buy_and_hold_returns),
    'CVaR': cvar(buy_and_hold_returns),
    'Max Drawdown': max_drawdown(buy_and_hold_returns)
}
# Print the performance metrics
print("Risk Parity Strategy Metrics:")
for metric, value in risk_parity_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nBuy and Hold Strategy Metrics:")
for metric, value in buy_and_hold_metrics.items():
    print(f"{metric}: {value:.4f}")
    
import matplotlib.pyplot as plt
# Calculate cumulative returns
risk_parity_cumulative_returns = (1 + risk_parity_returns).cumprod()
buy_and_hold_cumulative_returns = (1 + buy_and_hold_returns).cumprod()

# Plot the cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(risk_parity_cumulative_returns, label="Risk Parity Strategy")
plt.plot(buy_and_hold_cumulative_returns, label="Buy and Hold Strategy")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.title("Comparison of Cumulative Returns: Risk Parity vs Buy and Hold")
plt.show()


import scipy.stats as stats

def first_order_stochastic_dominance(returns1, returns2):
    cdf1 = stats.cumfreq(returns1, numbins=len(returns1))
    cdf2 = stats.cumfreq(returns2, numbins=len(returns2))

    for r1, r2 in zip(cdf1.cumcount, cdf2.cumcount):
        if r1 > r2:
            return False
    return True

def second_order_stochastic_dominance(returns1, returns2):
    cdf1 = stats.cumfreq(returns1, numbins=len(returns1))
    cdf2 = stats.cumfreq(returns2, numbins=len(returns2))

    int_cdf1 = np.trapz(cdf1.cumcount)
    int_cdf2 = np.trapz(cdf2.cumcount)

    for r1, r2 in zip(cdf1.cumcount, cdf2.cumcount):
        if r1 > r2:
            return False
    return int_cdf1 <= int_cdf2

def third_order_stochastic_dominance(returns1, returns2):
    cdf1 = stats.cumfreq(returns1, numbins=len(returns1))
    cdf2 = stats.cumfreq(returns2, numbins=len(returns2))

    int_cdf1 = np.trapz(cdf1.cumcount)
    int_cdf2 = np.trapz(cdf2.cumcount)

    int_cdf1_sq = np.trapz(np.square(cdf1.cumcount))
    int_cdf2_sq = np.trapz(np.square(cdf2.cumcount))

    for r1, r2 in zip(cdf1.cumcount, cdf2.cumcount):
        if r1 > r2:
            return False
    return (int_cdf1 <= int_cdf2) and (int_cdf1_sq <= int_cdf2_sq)


# Test stochastic dominance between risk parity and buy and hold strategies
print("\nRisk Parity vs Buy and Hold Stochastic Dominance Tests:")
first_order = first_order_stochastic_dominance(risk_parity_returns, buy_and_hold_returns)
print(f"1st Order Stochastic Dominance: {first_order}")

second_order = second_order_stochastic_dominance(risk_parity_returns, buy_and_hold_returns)
print(f"2nd Order Stochastic Dominance: {second_order}")

third_order = third_order_stochastic_dominance(risk_parity_returns, buy_and_hold_returns)
print(f"3rd Order Stochastic Dominance: {third_order}")