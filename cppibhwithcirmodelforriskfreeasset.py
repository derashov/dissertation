# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 16:48:18 2023

@author: danie
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def gbm(n_years = 10, n_scenarios=1000, mu=0.07, sigma1=0.15, steps_per_year=12, s_0=100.0, prices=True):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years: The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    # Derive per-step Model Parameters from User Specifications
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    # the standard way ...
    # rets_plus_1 = np.random.normal(loc=mu*dt+1, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    # without discretization error ...
    
    
  
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma1*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    ret_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
    return ret_val

def run_sim_cir(n_years=10, n_scenarios=1000, a=1, b=0.004, rt=0.1, sigma=0.09, steps_per_year=12):
    dt = 1 / steps_per_year
    n_steps = int(n_years * steps_per_year) + 1
    ret_valcir = np.random.normal(loc=0.0, scale=(np.sqrt(dt)), size=(n_steps, n_scenarios))
    rt = rt + a * (b - rt) * dt + sigma * np.sqrt(rt) * ret_valcir
    output = pd.DataFrame(rt)
    return output

# Run the simulation and get the results
sim_data = run_sim_cir(n_years=10, n_scenarios=50, b=0.004, rt=0.004, a = 0.3)

# Calculate the mean and standard deviation of the simulation results
mean_data = sim_data.mean(axis=1)
std_data = sim_data.std(axis=1)

# Create the figure and axes objects
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the mean line and the shaded area
ax.plot(mean_data, color='blue')
ax.fill_between(mean_data.index, mean_data - std_data, mean_data + std_data, color='blue', alpha=0.1)

# Set the axis labels and title
ax.set_xlabel('Month')
ax.set_ylabel('Interest Rate')
ax.set_title('CIR Model Simulation Results')

# Show the plot
plt.show()

def run_cppi_cir(risky_r, safe_r, m=3, start=1000, floor=0.8, a = 1, b = 0.1/12, rt = 0.1/12, sigma = 0.09, drawdown=None):
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
    total_trans_cost = 0.0
    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns=["R"])
    if isinstance(safe_r, pd.Series):
        safe_r = pd.DataFrame(safe_r, columns = ["Rs"]).reindex_like(risky_r)
    # set up some DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    floorval_history = pd.DataFrame().reindex_like(risky_r)
    peak_history = pd.DataFrame().reindex_like(risky_r)
    trans_cost = 0.0004
    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak*(1-drawdown)
        cushion = (account_value - floor_value)/account_value
        risky_w = m*cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1-risky_w
        if step > 0:
          trans_cost_step = trans_cost*(abs(risky_w - risky_w_history.iloc[step-1]) + abs(safe_w - (1-risky_w_history.iloc[step-1])))
          account_value *= (1-trans_cost_step)
          total_trans_cost += trans_cost_step
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w
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
        "floor": floorval_history,
        "transaction cost": total_trans_cost
    }
    return backtest_result

def show_cppi_cir(n_scenarios = 50, mu = 0.07, sigma1 = 0.15, m = 3, floor = 0.8, a = 1, b = 0.1/12, rt = 0.1/12, sigma = 0.09, y_max = 100):
    start = 100
    sim_rets = gbm(n_scenarios = n_scenarios, mu = mu, sigma1 = sigma1, prices = False, steps_per_year = 12)
    risky_r = pd.DataFrame(sim_rets)
    sim_rets1 = run_sim_cir(n_scenarios = n_scenarios, a = a, b = b, rt = rt, sigma = sigma, steps_per_year = 12)
    safe_r = pd.DataFrame(sim_rets1)
    btr = run_cppi_cir(risky_r = pd.DataFrame(risky_r), safe_r = pd.DataFrame(safe_r), m=m, start = start, floor = floor)
    wealth = btr["Wealth"]
    y_max = wealth.values.max()*y_max/100
    terminal_wealth = wealth.iloc[-1]
    transaction_cost= btr['transaction cost']
    
    tw_mean = terminal_wealth.mean()
    tw_median = terminal_wealth.median()
    failure_mask = np.less(terminal_wealth, start*floor)
    n_failures = failure_mask.sum()
    p_fail = n_failures/n_scenarios
    
    e_shortfall = np.dot(terminal_wealth-start*floor, failure_mask)/n_failures if n_failures > 0 else 0.0
    ann_ret = wealth.pct_change().mean()*12
    ann_ret = ann_ret.mean()
    total_ret = tw_mean/start - 1
    x = wealth.pct_change()-wealth.pct_change().mean()
    x = x.dropna()
    x2 = x**2
    row_sums = x2.sum()
    row_sums = row_sums/(120 - 1)
    volatility = np.sqrt(row_sums)
    volyear = volatility * np.sqrt(12)
    ann_vol = volyear.mean()
    
    min_value = terminal_wealth.min()
    max_value = terminal_wealth.max()
    max_drawdown = (wealth/wealth.cummax() - 1).min()
    max_drawdown1 = max_drawdown.min()
    skewness = terminal_wealth.skew()
    kurtosis = terminal_wealth.kurtosis()
    total_ret1 = (terminal_wealth-start)/start
    threshold = 0.0
    downside_returns = total_ret1[total_ret1 < threshold]
    downside_dev = np.std(downside_returns)
   
    
    sorted_ret1 = np.sort(total_ret1)
    var1 = sorted_ret1[int(0.05*n_scenarios)]
    
    
    tail_losses = sorted_ret1[sorted_ret1<var1]
    es1 = tail_losses.mean()
    sharpe_ratio = (ann_ret-0.05)/ann_vol
    sortino_ratio = (ann_ret-0.05)/downside_dev
    calmar_ratio = ann_ret/np.abs(max_drawdown1)
    fig, (wealth_ax, hist_ax) = plt.pyplot.subplots(nrows = 1, ncols = 2, sharey = True, gridspec_kw = {'width_ratios':[3,2]}, figsize = (24,9))
    plt.pyplot.subplots_adjust(wspace = 0.0)
    wealth.plot(ax = wealth_ax, legend = False, alpha = 0.3, color = 'indianred')
    wealth_ax.axhline(y = start, ls = ":", color = 'black')
    wealth_ax.axhline(y = start*floor, ls = '--', color = 'red')
    wealth_ax.set_ylim(top = y_max)
    terminal_wealth.plot.hist(ax = hist_ax, bins = 50, ec = 'w', fc = 'indianred', orientation = 'horizontal')
    hist_ax.axhline(y = start, ls = ':', color = 'black')
    
    hist_ax.axhline(y = start, ls = ':', color = 'black')
    hist_ax.axhline(y = tw_mean, ls = ':', color = 'blue')
    hist_ax.axhline(y = tw_median, ls = ':', color = 'purple')
    hist_ax.annotate(f"Mean: ₽{int(tw_mean)}", xy = (.5,.9), xycoords = 'axes fraction', fontsize = 24)
    hist_ax.annotate(f"Median: ₽{int(tw_median)}", xy = (.5,.85), xycoords = 'axes fraction', fontsize = 24)
    if (floor > 0.01):
        hist_ax.axhline(y = start*floor, ls = "--", color = "red", linewidth = 3)
        hist_ax.annotate(f"Violations: {n_failures} ({p_fail*100:2.2f}%)\nE(shortfall)=₽{e_shortfall:2.2f}", xy = (.5,.7),xycoords = 'axes fraction',fontsize = 24)
    print("annual volatility", ann_vol)
    print("annual return", ann_ret)
    print("total return", total_ret)
    print("minimal value", min_value)
    print("maximum value", max_value)
    print("max drawdown", max_drawdown1)
    print("skewness", skewness)
    print("kurtosis", kurtosis)
    print("downside deviation", downside_dev)
   
    print("VaR", var1)
    print("es", es1)
    print("sharpe ratio", sharpe_ratio)
    print("sortino ratio", sortino_ratio)
    print("calmar ratio", calmar_ratio)
    print('average transaction cost per simulation', transaction_cost.mean())
    df = pd.DataFrame(wealth)
    
    df.to_excel('twcppi.xlsx', index=False)     
import matplotlib as plt    
show_cppi_cir(10000, mu = 0.12, sigma1 =0.28, m = 3, floor = 0.8, b = 0.004, rt = 0.004)



def run_buy_and_hold(risky_r, safe_r, a = 1, b = 0.1/12, rt = 0.1/12, sigma = 0.09, start=100):
    """
    Run a backtest of the Buy and Hold strategy, given a set of returns for the risky asset and the risk-free asset
    Returns a dictionary containing: Asset Value History, Risky Weight History, Risk-Free Weight History
    """
    # set up the Buy and Hold parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns=["R"])
    if isinstance(safe_r, pd.Series):
        safe_r = pd.DataFrame(safe_r, columns = ["Rs"]).reindex_like(risky_r)
    # set up some DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    safe_w_history = pd.DataFrame().reindex_like(risky_r)
    
    for step in range(n_steps):
        risky_w = 0.6
        safe_w = 0.4
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w
        # recompute the new account value at the end of this step
        account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])
        # save the histories for analysis and plotting
        risky_w_history.iloc[step] = risky_w
        safe_w_history.iloc[step] = safe_w
        account_history.iloc[step] = account_value
    
    risky_wealth = start*(1+risky_r).cumprod()
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risky Allocation": risky_w_history,
        "Risk-Free Allocation": safe_w_history,
        "start": start,
        "risky_r":risky_r,
        "safe_r": safe_r
    }
    return backtest_result

from scipy.stats import norm

def show_buy_and_hold(n_scenarios = 50, mu = 0.07, sigma1 = 0.15, a = 1, b = 0.1/12, rt = 0.1/12, sigma = 0.09, y_max = 100):
    start = 100
    sim_rets = gbm(n_scenarios = n_scenarios, mu = mu, sigma1 = sigma1, prices = False, steps_per_year = 12)
    risky_r = pd.DataFrame(sim_rets)
    sim_rets1 = run_sim_cir(n_scenarios = n_scenarios, a = a, b = b, rt = rt, sigma = sigma, steps_per_year = 12)
    safe_r = pd.DataFrame(sim_rets1)
    btr = run_buy_and_hold(risky_r = pd.DataFrame(risky_r), safe_r = pd.DataFrame(safe_r), start = start)
    wealth = btr["Wealth"]
    y_max = wealth.values.max()*y_max/100
    terminal_wealth = wealth.iloc[-1]
    
    tw_mean = terminal_wealth.mean()
    tw_median = terminal_wealth.median()
    
    
   
    
    ann_ret = wealth.pct_change().mean()*12
    ann_ret = ann_ret.mean()
    total_ret = tw_mean/start - 1
    x = wealth.pct_change()-wealth.pct_change().mean()
    x = x.dropna()
    x2 = x**2
    row_sums = x2.sum()
    row_sums = row_sums/(120 - 1)
    volatility = np.sqrt(row_sums)
    volyear = volatility * np.sqrt(12)
    ann_vol = volyear.mean()
    
    min_value = terminal_wealth.min()
    max_value = terminal_wealth.max()
    max_drawdown = (wealth/wealth.cummax() - 1).min()
    max_drawdown1 = max_drawdown.min()
    skewness = terminal_wealth.skew()
    kurtosis = terminal_wealth.kurtosis()
    total_ret1 = (terminal_wealth-start)/start
    threshold = 0.0
    downside_returns = total_ret1[total_ret1 < threshold]
    downside_dev = np.std(downside_returns)
   
    
    sorted_ret1 = np.sort(total_ret1)
    var1 = sorted_ret1[int(0.05*n_scenarios)]
    
    
    tail_losses = sorted_ret1[sorted_ret1<var1]
    es1 = tail_losses.mean()
    sharpe_ratio = (ann_ret-0.05)/ann_vol
    sortino_ratio = (ann_ret-0.05)/downside_dev
    calmar_ratio = ann_ret/np.abs(max_drawdown1)
   
    
    fig, (wealth_ax, hist_ax) = plt.pyplot.subplots(nrows = 1, ncols = 2, sharey = True, gridspec_kw = {'width_ratios':[3,2]}, figsize = (24,9))
    plt.pyplot.subplots_adjust(wspace = 0.0)
    wealth.plot(ax = wealth_ax, legend = False, alpha = 0.3, color = 'indianred')
    wealth_ax.axhline(y = start, ls = ":", color = 'black')
    
    wealth_ax.set_ylim(top = y_max)
    terminal_wealth.plot.hist(ax = hist_ax, bins = 50, ec = 'w', fc = 'indianred', orientation = 'horizontal')
    hist_ax.axhline(y = start, ls = ':', color = 'black')

    hist_ax.axhline(y = start, ls = ':', color = 'black')
    hist_ax.axhline(y = tw_mean, ls = ':', color = 'blue')
    hist_ax.axhline(y = tw_median, ls = ':', color = 'purple')
    
    hist_ax.annotate(f"Mean: ₽{int(tw_mean)}", xy = (.5,.9), xycoords = 'axes fraction', fontsize = 24)
    hist_ax.annotate(f"Median: ₽{int(tw_median)}", xy = (.5,.85), xycoords = 'axes fraction', fontsize = 24)
    
    print("annual volatility", ann_vol)
    print("annual return", ann_ret)
    print("total return", total_ret)
    print("minimal value", min_value)
    print("maximum value", max_value)
    print("max drawdown", max_drawdown1)
    print("skewness", skewness)
    print("kurtosis", kurtosis)
    print("downside deviation", downside_dev)
   
    print("VaR", var1)
    print("es", es1)
    print("sharpe ratio", sharpe_ratio)
    print("sortino ratio", sortino_ratio)
    print("calmar ratio", calmar_ratio)
    df = pd.DataFrame(wealth)
    df.to_excel('twbh.xlsx', index=False)

show_buy_and_hold(10000, mu = 0.12, sigma1 =0.28, b = 0.004, rt = 0.004)

dfbh = pd.read_excel('twbh.xlsx')

tw = dfbh.tail(1).T
tw = pd.DataFrame(tw)

dfcppi = pd.read_excel('twcppi.xlsx')

twcppi = dfcppi.tail(1).T
twcppi = pd.DataFrame(twcppi)

sorted_values_1 = np.sort(tw[120])
sorted_values_2 = np.sort(twcppi[120])

n_simulations = 10000

n_pairs = n_simulations * (n_simulations - 1) / 2
count_3 = 0
count_4 = 0
count_5 = 0
for i in range(n_simulations):
    for j in range(i+1, n_simulations):
        if sorted_values_1[i] + sorted_values_1[j] < 2*sorted_values_2[i]:
            count_3 += 1
            if sorted_values_1[i] + sorted_values_1[j] + sorted_values_1[i] < 3*sorted_values_2[i]:
                count_4 += 1
                if sorted_values_1[i] + sorted_values_1[j] + sorted_values_1[i] + sorted_values_1[j] < 4*sorted_values_2[i]:
                    count_5 += 1

pct_3 = count_3 / n_pairs
pct_4 = count_4 / n_pairs
pct_5 = count_5 / n_pairs

# Check if TOSD condition is satisfied for any order
alpha = 0.05
if pct_3 > alpha:
    print("TOSD at third order")
elif pct_4 > alpha:
    print("TOSD at fourth order")
elif pct_5 > alpha:
    print("TOSD at fifth order")
else:
    print("No TOSD detected")