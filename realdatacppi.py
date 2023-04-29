# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 23:56:56 2023

@author: danie
"""

import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt

risk_asset_file = 'risky_asset_data.csv'
risk_asset_returns = pd.read_csv(risk_asset_file, index_col=0)
risk_asset_returns = risk_asset_returns.rename(columns = {'price':'MOEX'})


risk_free_asset_file = 'risk_free_asset_data.csv'
risk_free_asset_returns = pd.read_csv(risk_free_asset_file, index_col=0)
risk_free_asset_returns = risk_free_asset_returns.rename(columns = {'price':'RGBITOR'})

risky = risk_asset_returns.pct_change().dropna()
safe = risk_free_asset_returns.pct_change().dropna()

def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, drawdown=None, freq=None):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
    """
    # set up the CPPI parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor
    peak = account_value
    if isinstance(risky_r, pd.Series): 
        risky_r = pd.DataFrame(risky_r, columns=["R"])

    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/350 # fast way to set all values to a number
    else:
        o = safe_r
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.loc[:,:] = o.values.reshape(-1, 1)
    
    # set up some DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    floorval_history = pd.DataFrame().reindex_like(risky_r)
    peak_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak*(1-drawdown)
        if freq is not None and step % freq == 0:
            cushion = (account_value - floor_value)/account_value
            risky_w = m*cushion
            risky_w = np.minimum(risky_w, 1)
            risky_w = np.maximum(risky_w, 0)
            safe_w = 1-risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w
        # recompute the new account value at the end of this step
        account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])
       
        # save the histories for analysis and plotting
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
        floorval_history.iloc[step] = floor_value
        peak_history.iloc[step] = peak
    risky_wealth = start*(1+risky_r).cumprod()
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth, 
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r":risky_r,
        "safe_r": safe_r,
        "drawdown": drawdown,
        "peak": peak_history,
        "floor": floorval_history
    }
    return backtest_result, safe_r


def summary_stats(backtest_result):
    wealth = backtest_result['Wealth']
    
    returns = wealth.pct_change().dropna()
    std = returns.std()*np.sqrt(252)
    cagr = (wealth.iloc[-1]/wealth.iloc[0])**(252/len(wealth)) - 1
    riskfree_rate = 0.06

    sharpe = (cagr-riskfree_rate)/std   
    max_dd = ((wealth / wealth.cummax()) - 1).min()
    summary = {
        'CAGR': cagr,
        
        'Volatility': std,
        
        
        'Sharpe Ratio': sharpe,
        
        'Max Drawdown': max_dd,
        
    }

    return summary





btr = run_cppi(risky['2015':], safe['2015':], m=3, start=100, floor=0.8, drawdown=0.1, freq=30)
ax = btr[0]["Wealth"].plot(figsize=(15,6))
btr[0]["Risky Wealth"].plot(ax=ax, style="--", linewidth=3)

summary = summary_stats(btr[0])
print(summary)



initial_weights = [0.6, 0.4]
portfolio = pd.concat([risky, safe], axis=1)
portfolio = portfolio.fillna(method='ffill').dropna()

# Compute the portfolio returns
portfolio_returns = portfolio.dot(initial_weights)

# Compute the portfolio value over time
portfolio_value = 100 * (1 + portfolio_returns).cumprod()

# Plot the portfolio value over time
portfolio_value.plot(figsize=(15,6))
plt.title('60-40 Portfolio Buy and Hold Strategy')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.show()

# Compute and print summary statistics
returns = portfolio_value.pct_change().dropna()
std = returns.std() * np.sqrt(252)
cagr = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (252 / len(portfolio_value)) - 1
sharpe = (cagr - 0.03) / std
max_dd = ((portfolio_value / portfolio_value.cummax()) - 1).min()
summary = {
    'CAGR': cagr,
    'Volatility': std,
    'Sharpe Ratio': sharpe,
    'Max Drawdown': max_dd
}
print(summary)





btr = run_cppi(risky['2015':], safe['2015':], m=3, start=100, floor=0.8, freq=30)
ax = btr[0]["Wealth"].plot(figsize=(15,6))


initial_weights = [0.6, 0.4]
portfolio = pd.concat([risky, safe], axis=1)
portfolio = portfolio.fillna(method='ffill').dropna()

# Compute the portfolio returns
portfolio_returns = portfolio.dot(initial_weights)

# Compute the portfolio value over time
portfolio_value = 100 * (1 + portfolio_returns).cumprod()

# Plot the portfolio value over time on the same axis
portfolio_value.plot(ax=ax, style="-", linewidth=3)

plt.title('CPPI vs. 60-40 Portfolio Buy and Hold Strategy')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend(['CPPI', 'Buy and Hold'])
plt.show()

# Compute and print summary statistics for both strategies
cppi_summary = summary_stats(btr[0])
print("CPPI summary:")
print(cppi_summary)

portfolio_summary = {
    'CAGR': cagr,
    'Volatility': std,
    'Sharpe Ratio': sharpe,
    'Max Drawdown': max_dd
}
print("Portfolio summary:")
print(portfolio_summary)


# Set style for the plot
plt.style.use('seaborn')

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)

# Plot the CPPI and risky asset wealth on the first subplot
ax1.plot(btr[0]["Wealth"], label='CPPI', linewidth=2)
ax1.plot(btr[0]["Risky Wealth"], label='Risky Asset', linestyle='--', linewidth=2)
ax1.set_ylabel('Portfolio Value')
ax1.legend()

# Plot the 60-40 portfolio on the second subplot
ax2.plot(portfolio_value, label='60-40 Portfolio', linewidth=2)
ax2.plot(btr[0]["Risky Wealth"], label='Risky Asset', linestyle='--', linewidth=2)
ax2.set_xlabel('Date')
ax2.set_ylabel('Portfolio Value')
ax2.legend()

# Set title and adjust spacing
plt.suptitle('Comparison of CPPI and 60-40 Portfolio', fontsize=16, fontweight='bold')
plt.subplots_adjust(top=0.9, hspace=0.1)

# Show the plot
plt.show()


import matplotlib.dates as mdates

# set the date formatter
date_format = mdates.DateFormatter('%Y-%m-%d')

# plot the graph
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(btr[0]["Wealth"].index, btr[0]["Wealth"], label="CPPI", linewidth=2)
ax.plot(btr[0]["Risky Wealth"].index, btr[0]["Risky Wealth"], label="Risky Wealth", linestyle="--", linewidth=2)
ax.plot(portfolio_value.index, portfolio_value, label="60-40 Portfolio", linewidth=2)
ax.legend(loc="upper left")

# set the x-axis label and title
ax.set_xlabel("Date")
ax.set_title("Comparison of CPPI and 60-40 Portfolio Strategies")

# set the date formatter for the x-axis
ax.xaxis.set_major_formatter(date_format)

plt.show()

