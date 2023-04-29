import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read CSV files
risky_asset = pd.read_csv('risky_asset_data.csv')
risk_free_asset = pd.read_csv('risk_free_asset_data.csv')

# Use the same date range for both assets
merged_data = pd.merge(risky_asset, risk_free_asset, on='date', suffixes=('_risky', '_risk_free'))
merged_data['date'] = pd.to_datetime(merged_data['date'])
merged_data.set_index('date', inplace=True)

# Restrict date range
start_date = '2017-01-30'
end_date = '2020-12-30'

risky_asset['date'] = pd.to_datetime(risky_asset['date'])
risk_free_asset['date'] = pd.to_datetime(risk_free_asset['date'])

risky_asset = risky_asset[(risky_asset['date'] >= start_date) & (risky_asset['date'] <= end_date)]
risk_free_asset = risk_free_asset[(risk_free_asset['date'] >= start_date) & (risk_free_asset['date'] <= end_date)]

# Calculate the rate of change (ROC) over a specific lookback period
lookback_period = 63  # Approximately 6 months

merged_data['ROC_risky'] = merged_data['price_risky'].pct_change(periods=lookback_period)
merged_data['ROC_risk_free'] = merged_data['price_risk_free'].pct_change(periods=lookback_period)

# Calculate the weight allocation based on the momentum of the risky asset relative to the risk-free asset
merged_data['Momentum'] = merged_data['ROC_risky'] - merged_data['ROC_risk_free']
merged_data['Signal'] = np.clip(merged_data['Momentum'], 0, 1)

#Calculate short and long moving averages
short_window = 50
long_window = 250
merged_data['Short_MA'] = merged_data['price_risky'].rolling(window=short_window).mean()
merged_data['Long_MA'] = merged_data['price_risky'].rolling(window=long_window).mean()

#Create signals based on moving average crossovers
merged_data['MA_Signal'] = np.where(merged_data['Short_MA'] > merged_data['Long_MA'], 1, 0)


#Calculate daily returns
merged_data['Return_risky'] = merged_data['price_risky'].pct_change()
merged_data['Return_risk_free'] = merged_data['price_risk_free'].pct_change()

#Calculate strategy returns with transaction costs
transaction_cost = 0.0004 # 0.04% per transaction
merged_data['Transaction_cost'] = np.abs(merged_data['Signal'].diff()) * transaction_cost
merged_data['TAA_Strategy'] = (merged_data['Signal'].shift(1) * merged_data['Return_risky'] + (1 - merged_data['Signal'].shift(1)) * merged_data['Return_risk_free']) * (1 - merged_data['Transaction_cost'])

#Calculate TAA_MA_Strategy returns with transaction costs

merged_data['Transaction_cost_MA'] = merged_data['MA_Signal'].shift(1) != merged_data['MA_Signal']
merged_data['Transaction_cost_MA'] = merged_data['Transaction_cost_MA'].astype(int) * transaction_cost
merged_data['TAA_MA_Strategy'] = (merged_data['MA_Signal'].shift(1) * merged_data['Return_risky'] + (1 - merged_data['MA_Signal'].shift(1)) * merged_data['Return_risk_free']) * (1 - merged_data['Transaction_cost_MA'])

#Set start date to be the same for all strategies
start_date = merged_data.index[max(lookback_period, long_window)]

#Filter data to start from the same date
filtered_data = merged_data.loc[merged_data.index >= start_date].copy(deep=True)

filtered_data['Transaction_cost_MA'] = np.abs(filtered_data['MA_Signal'].diff()) * transaction_cost
filtered_data['TAA_MA_Strategy'] = (filtered_data['MA_Signal'].shift(1) * filtered_data['Return_risky'] + (1 - filtered_data['MA_Signal'].shift(1)) * filtered_data['Return_risk_free']) * (1 - filtered_data['Transaction_cost_MA'])
filtered_data['MA_Signal'].iloc[0]= 0.2
filtered_data['TAA_MA_Strategy'] = 100 * (1 + filtered_data['TAA_MA_Strategy']).cumprod()

#Calculate Buy-and-Hold returns with fixed weights starting from the same date
filtered_data['Buy_and_Hold'] = 0.2 * filtered_data['Return_risky'] + 0.8 * filtered_data['Return_risk_free']
filtered_data['Buy_and_Hold'] = 100 * (1 + filtered_data['Buy_and_Hold']).cumprod()

#Calculate cumulative returns starting from an initial value of 100
filtered_data['Signal'].iloc[0]= 0.2
filtered_data['Adjusted_Signal'] = 0.2 + filtered_data['Momentum']
filtered_data['Adjusted_Signal'] = np.clip(filtered_data['Adjusted_Signal'], 0, 1)
filtered_data['TAA_Strategy'] = (filtered_data['Adjusted_Signal'].shift(1) * filtered_data['Return_risky'] + (1 - filtered_data['Adjusted_Signal'].shift(1)) * filtered_data['Return_risk_free']) * (1 - filtered_data['Transaction_cost'])
filtered_data['TAA_Strategy'] = 100 * (1 + filtered_data['TAA_Strategy']).cumprod()





#Plot cumulative returns for the filtered data
filtered_data[['TAA_Strategy', 'TAA_MA_Strategy', 'Buy_and_Hold']].plot(figsize=(12, 8))
plt.title('TAA Strategies vs. Buy-and-Hold (From ' + str(start_date) + ')')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend(['TAA Strategy (Momentum)', 'TAA Strategy (MA)', 'Buy-and-Hold (20% Risky, 80% Risk-free)'])
plt.show()

#Calculate portfolio statistics
trading_days = 252

def calculate_statistics(returns):
    annualized_return = (1 + returns.mean()) ** trading_days - 1
    annualized_volatility = returns.std() * np.sqrt(trading_days)
    sharpe_ratio = annualized_return / annualized_volatility
    return annualized_return, annualized_volatility, sharpe_ratio

annualized_return_taa, annualized_volatility_taa, sharpe_ratio_taa = calculate_statistics(filtered_data['TAA_Strategy'].pct_change().dropna())
annualized_return_taa_ma, annualized_volatility_taa_ma, sharpe_ratio_taa_ma = calculate_statistics(filtered_data['TAA_MA_Strategy'].pct_change().dropna())
annualized_return_bh, annualized_volatility_bh, sharpe_ratio_bh = calculate_statistics(filtered_data['Buy_and_Hold'].pct_change().dropna())

#Calculate total transaction costs for the TAA strategies
total_transaction_costs = (filtered_data['Transaction_cost'] * filtered_data['TAA_Strategy'].shift(1)).sum()
total_transaction_costs_ma = (filtered_data['Transaction_cost_MA'] * filtered_data['TAA_MA_Strategy'].shift(1)).sum()

#Calculate Value at Risk (VaR), Conditional Value at Risk (CVaR), and Maximum Drawdown
def calculate_risk_metrics(returns):
    VaR_95 = returns.quantile(0.05)
    VaR_95_annual = VaR_95 * np.sqrt(252)
    CVaR_95 = returns[returns <= VaR_95].mean() * np.sqrt(252)
    cumulative_returns = (1 + returns).cumprod()
    max_value = cumulative_returns.cummax()
    drawdown = (cumulative_returns - max_value) / max_value
    max_drawdown = drawdown.min()
    
    return VaR_95_annual, CVaR_95, max_drawdown

VaR_95_taa_ma, CVaR_95_taa_ma, max_drawdown_taa_ma = calculate_risk_metrics(filtered_data['TAA_MA_Strategy'].pct_change().dropna())

VaR_95_bh, CVaR_95_bh, max_drawdown_bh = calculate_risk_metrics(filtered_data['Buy_and_Hold'].pct_change().dropna())

VaR_95_taa, CVaR_95_taa, max_drawdown_taa = calculate_risk_metrics(filtered_data['TAA_Strategy'].pct_change().dropna())

sum_nonzero = (filtered_data['Transaction_cost'] != 0).sum()

sum_nonzero1 = (filtered_data['Transaction_cost_MA'] != 0).sum()
print(sum_nonzero)
print(sum_nonzero1)

#Print portfolio statistics
print("TAA Strategy (Momentum):")
print(f"Annualized Return: {annualized_return_taa:.2%}")
print(f"Annualized Volatility: {annualized_volatility_taa:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio_taa:.2f}")
print(f"Total Transaction Costs: {total_transaction_costs:.2f}")
print(f"Value at Risk (95%): {VaR_95_taa:.2%}")
print(f"Conditional Value at Risk (95%): {CVaR_95_taa:.2%}")
print(f"Maximum Drawdown: {max_drawdown_taa:.2%}")

print("\nTAA Strategy (Moving Averages):")
print(f"Annualized Return: {annualized_return_taa_ma:.2%}")
print(f"Annualized Volatility: {annualized_volatility_taa_ma:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio_taa_ma:.2f}")
print(f"Total Transaction Costs: {total_transaction_costs_ma:.2f}")
print(f"Value at Risk (95%): {VaR_95_taa_ma:.2%}")
print(f"Conditional Value at Risk (95%):{CVaR_95_taa_ma:.2%}")
print(f"Maximum Drawdown: {max_drawdown_taa_ma:.2%}")

print("\nBuy-and-Hold Strategy (60% Risky, 40% Risk-free):")
print(f"Annualized Return: {annualized_return_bh:.2%}")
print(f"Annualized Volatility: {annualized_volatility_bh:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio_bh:.2f}")
print(f"Value at Risk (95%): {VaR_95_bh:.2%}")
print(f"Conditional Value at Risk (95%): {CVaR_95_bh:.2%}")
print(f"Maximum Drawdown: {max_drawdown_bh:.2%}")

from scipy.stats import rankdata
from itertools import permutations

# Define the 3-level stochastic dominance function
def three_level_stochastic_dominance(data):
    n = data.shape[0]
    if n <= 1:
        return False
    for i in range(1, n+1):
        for j in range(1, n+1):
            if i != j:
                for k in range(1, n+1):
                    if k != i and k != j:
                        perm = permutations([i, j, k])
                        for p in perm:
                            w1, w2, w3 = p
                            if np.sum(data[:,w1-1] > data[:,w2-1]) >= np.ceil(n/2) and np.sum(data[:,w2-1] > data[:,w3-1]) >= np.ceil(n/2) and np.sum(data[:,w1-1] > data[:,w3-1]) >= np.ceil(n/2):
                                return True
    return False

returns = np.column_stack((filtered_data['TAA_Strategy'], filtered_data['TAA_MA_Strategy'], filtered_data['Buy_and_Hold']))
ranks = np.apply_along_axis(rankdata, 0, returns)
if three_level_stochastic_dominance(ranks):
    print("Three-level stochastic dominance is present")
else:
    print("Three-level stochastic dominance is not present")
    
# Calculate the rank-based dominance matrix
n_strategies = returns.shape[1]
dominance_matrix = np.zeros((n_strategies, n_strategies))
for i in range(n_strategies):
    for j in range(i+1, n_strategies):
        dominance_count = 0
        for k in range(n_strategies):
            if ranks[k,i] > ranks[k,j]:
                dominance_count += 1
        if dominance_count >= np.ceil(n_strategies/3)*2:
            dominance_matrix[i,j] = 1
        elif dominance_count <= np.floor(n_strategies/3):
            dominance_matrix[j,i] = 1

# Print the dominance matrix
print("Dominance Matrix:")
print(dominance_matrix)

# Determine which strategy dominates
dominant_strategy = np.sum(dominance_matrix, axis=1)
if np.sum(dominant_strategy == n_strategies-1) == 1:
    idx = np.where(dominant_strategy == n_strategies-1)[0][0]
    print("Strategy", idx+1, "dominates all other strategies")
else:
    print("No dominant strategy found")
    

# Calculate annual returns
annual_returns_taa = (filtered_data['TAA_Strategy'].pct_change().dropna() + 1).resample('Y').prod() - 1
annual_returns_taa_ma = (filtered_data['TAA_MA_Strategy'].pct_change().dropna() + 1).resample('Y').prod() - 1
annual_returns_bh = (filtered_data['Buy_and_Hold'].pct_change().dropna() + 1).resample('Y').prod() - 1


# Perform two-sample t-test
ttest_taa_taa_ma = ttest_ind(annual_returns_taa, annual_returns_taa_ma)
ttest_taa_bh = ttest_ind(annual_returns_taa, annual_returns_bh)
ttest_taa_ma_bh = ttest_ind(annual_returns_taa_ma, annual_returns_bh)

print("\nTwo-sample t-test results:")
print("TAA vs. TAA_MA:", ttest_taa_taa_ma)
print("TAA vs. Buy-and-Hold:", ttest_taa_bh)
print("TAA_MA vs. Buy-and-Hold:", ttest_taa_ma_bh)

# Perform Levene's test for equal variances
levene_test_taa_taa_ma = levene(annual_returns_taa, annual_returns_taa_ma)
levene_test_taa_bh = levene(annual_returns_taa, annual_returns_bh)
levene_test_taa_ma_bh = levene(annual_returns_taa_ma, annual_returns_bh)

print("\nLevene's test for equal variances results:")
print("TAA vs. TAA_MA:", levene_test_taa_taa_ma)
print("TAA vs. Buy-and-Hold:", levene_test_taa_bh)
print("TAA_MA vs. Buy-and-Hold:", levene_test_taa_ma_bh)