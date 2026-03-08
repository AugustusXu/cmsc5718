"""
CMSC 5718 Assignment 1: Derivative Pricing and Hedging
Student number last digit: 5
S1: China Construction Bank (939), S0=7.92, sigma=0.202
S2: Agricultural Bank (1288), S0=5.60, sigma=0.232
Correlation: 0.621
Number of options M = 250,000
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import time

np.random.seed(42)

# ==============================================================================
# Common Parameters
# ==============================================================================
r = 0.0314          # continuously compounded interest rate
S1_0 = 7.92         # initial price of S1 (CCB 939) as of June 30, 2025
S2_0 = 5.60         # initial price of S2 (Agricultural Bank 1288)
sigma1 = 0.202      # volatility of S1
sigma2 = 0.232      # volatility of S2
rho = 0.621         # correlation coefficient
M = 250000          # number of options
K_option = S1_0     # ATM strike price (S = K = 7.92)

print("=" * 80)
print("CMSC 5718 Assignment 1 - Student Number Ending in 5")
print("=" * 80)
print(f"S1: China Construction Bank (939), S0 = {S1_0}")
print(f"S2: Agricultural Bank (1288), S0 = {S2_0}")
print(f"sigma1 = {sigma1}, sigma2 = {sigma2}, rho = {rho}")
print(f"r = {r}, M = {M}")
print()

# ==============================================================================
# Question 1: European Option Pricing (20%)
# ==============================================================================
print("=" * 80)
print("Question 1: European Call Option Pricing")
print("=" * 80)

# ---------- Q1(i): Black-Scholes Analytical Price ----------
T1 = 0.5014
S = S1_0
K = K_option

d1 = (np.log(S / K) + (r + 0.5 * sigma1**2) * T1) / (sigma1 * np.sqrt(T1))
d2 = d1 - sigma1 * np.sqrt(T1)

bs_price = S * norm.cdf(d1) - K * np.exp(-r * T1) * norm.cdf(d2)

print(f"\nQ1(i) Black-Scholes Analytical Price")
print(f"  T = {T1}, S = K = {K}, r = {r}, sigma = {sigma1}")
print(f"  d1 = {d1:.6f}")
print(f"  d2 = {d2:.6f}")
print(f"  N(d1) = {norm.cdf(d1):.6f}")
print(f"  N(d2) = {norm.cdf(d2):.6f}")
print(f"  BS Call Price = {bs_price:.6f}")
print()

# ---------- Q1(ii): Monte Carlo Pricing ----------
N_steps = 120
dt = T1 / N_steps

def monte_carlo_european_call(S0, K, r, sigma, T, N_steps, n_paths):
    dt = T / N_steps
    S = np.full(n_paths, S0, dtype=np.float64)
    for _ in range(N_steps):
        Z = np.random.standard_normal(n_paths)
        S = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    payoff = np.maximum(S - K, 0)
    price = np.exp(-r * T) * np.mean(payoff)
    std_err = np.exp(-r * T) * np.std(payoff) / np.sqrt(n_paths)
    return price, std_err

# (a) 10,000 paths
np.random.seed(42)
start_time = time.time()
mc_price_10k, mc_se_10k = monte_carlo_european_call(S, K, r, sigma1, T1, N_steps, 10000)
time_10k = time.time() - start_time

print(f"Q1(ii)(a) Monte Carlo with 10,000 paths")
print(f"  Time steps N = {N_steps}, dt = {dt:.6f}")
print(f"  MC Price = {mc_price_10k:.6f}")
print(f"  Std Error = {mc_se_10k:.6f}")
print(f"  Computation time = {time_10k:.4f} seconds")
print()

# (b) 500,000 paths
np.random.seed(42)
start_time = time.time()
mc_price_500k, mc_se_500k = monte_carlo_european_call(S, K, r, sigma1, T1, N_steps, 500000)
time_500k = time.time() - start_time

print(f"Q1(ii)(b) Monte Carlo with 500,000 paths")
print(f"  Time steps N = {N_steps}, dt = {dt:.6f}")
print(f"  MC Price = {mc_price_500k:.6f}")
print(f"  Std Error = {mc_se_500k:.6f}")
print(f"  Computation time = {time_500k:.4f} seconds")
print()

print(f"  Comparison: BS Price = {bs_price:.6f}")
print(f"              MC 10k   = {mc_price_10k:.6f} (diff = {abs(mc_price_10k - bs_price):.6f})")
print(f"              MC 500k  = {mc_price_500k:.6f} (diff = {abs(mc_price_500k - bs_price):.6f})")
print()

# ==============================================================================
# Question 2: Exotic Option Pricing (35%)
# ==============================================================================
print("=" * 80)
print("Question 2: Exotic Option Pricing")
print("=" * 80)

T2 = 0.5
n_paths_q2 = 1000000

def exotic_option_price(K_exotic, S1_0, S2_0, sigma1, sigma2, rho, r, T, n_paths):
    Z1 = np.random.standard_normal(n_paths)
    Z2 = np.random.standard_normal(n_paths)
    W1 = Z1
    W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

    S1_T = S1_0 * np.exp((r - 0.5 * sigma1**2) * T + sigma1 * np.sqrt(T) * W1)
    S2_T = S2_0 * np.exp((r - 0.5 * sigma2**2) * T + sigma2 * np.sqrt(T) * W2)

    B = np.minimum(S1_T / S1_0, S2_T / S2_0)

    payoff = np.where(
        (B >= 1) & (B <= 1 + K_exotic),
        K_exotic,
        B - 1
    )

    price = np.exp(-r * T) * np.mean(payoff)
    return price

def price_minus_target(K_exotic, target=-0.03):
    np.random.seed(123)
    price = exotic_option_price(K_exotic, S1_0, S2_0, sigma1, sigma2, rho, r, T2, n_paths_q2)
    return price - target

print("\nScanning K values to find the root bracket...")
K_test_values = np.arange(0.01, 1.0, 0.02)
prices_test = []
for k_val in K_test_values:
    np.random.seed(123)
    p = exotic_option_price(k_val, S1_0, S2_0, sigma1, sigma2, rho, r, T2, n_paths_q2)
    prices_test.append(p)
    if abs(p - (-0.03)) < 0.005:
        print(f"  K = {k_val:.4f}, Price = {p:.6f}")

for i in range(len(prices_test) - 1):
    if (prices_test[i] - (-0.03)) * (prices_test[i + 1] - (-0.03)) < 0:
        K_low, K_high = K_test_values[i], K_test_values[i + 1]
        print(f"  Root bracket found: K in [{K_low:.4f}, {K_high:.4f}]")
        break

K_solution = brentq(price_minus_target, K_low, K_high, xtol=1e-6)

np.random.seed(123)
final_price = exotic_option_price(K_solution, S1_0, S2_0, sigma1, sigma2, rho, r, T2, n_paths_q2)

print(f"\nQ2 Result:")
print(f"  K = {K_solution:.6f} ({K_solution*100:.4f}%)")
print(f"  Fair price at K = {K_solution:.6f} is {final_price:.6f}")
print(f"  Target price = -0.03")
print(f"  Difference = {abs(final_price - (-0.03)):.8f}")
print()

# ==============================================================================
# Question 3: Delta Hedging Strategy (45%)
# ==============================================================================
print("=" * 80)
print("Question 3: Delta Hedging Strategy")
print("=" * 80)

price_data = [
    ("2025-06-30", 7.92),
    ("2025-07-02", 8.15),
    ("2025-07-03", 8.15),
    ("2025-07-04", 8.21),
    ("2025-07-07", 8.20),
    ("2025-07-08", 8.24),
    ("2025-07-09", 8.22),
    ("2025-07-10", 8.48),
    ("2025-07-11", 8.35),
    ("2025-07-14", 8.41),
    ("2025-07-15", 8.42),
    ("2025-07-16", 8.39),
    ("2025-07-17", 8.31),
    ("2025-07-18", 8.47),
    ("2025-07-21", 8.37),
    ("2025-07-22", 8.27),
    ("2025-07-23", 8.38),
    ("2025-07-24", 8.42),
    ("2025-07-25", 8.30),
    ("2025-07-28", 8.34),
    ("2025-07-29", 8.19),
    ("2025-07-30", 8.19),
    ("2025-07-31", 8.05),
    ("2025-08-01", 7.89),
    ("2025-08-04", 7.92),
    ("2025-08-05", 8.00),
    ("2025-08-06", 7.93),
    ("2025-08-07", 8.03),
    ("2025-08-08", 7.94),
    ("2025-08-11", 7.93),
    ("2025-08-12", 7.95),
    ("2025-08-13", 8.02),
    ("2025-08-14", 7.98),
    ("2025-08-15", 7.80),
    ("2025-08-18", 7.71),
    ("2025-08-19", 7.71),
    ("2025-08-20", 7.75),
    ("2025-08-21", 7.74),
    ("2025-08-22", 7.75),
    ("2025-08-25", 7.73),
    ("2025-08-26", 7.55),
    ("2025-08-27", 7.49),
    ("2025-08-28", 7.53),
    ("2025-08-29", 7.51),
    ("2025-09-01", 7.55),
    ("2025-09-02", 7.70),
    ("2025-09-03", 7.63),
    ("2025-09-04", 7.59),
    ("2025-09-05", 7.67),
    ("2025-09-08", 7.63),
    ("2025-09-09", 7.75),
    ("2025-09-10", 7.97),
    ("2025-09-11", 7.88),
    ("2025-09-12", 7.88),
    ("2025-09-15", 7.75),
    ("2025-09-16", 7.77),
    ("2025-09-17", 7.84),
    ("2025-09-18", 7.65),
    ("2025-09-19", 7.61),
    ("2025-09-22", 7.45),
    ("2025-09-23", 7.50),
    ("2025-09-24", 7.47),
    ("2025-09-25", 7.30),
    ("2025-09-26", 7.38),
    ("2025-09-29", 7.48),
    ("2025-09-30", 7.48),
    ("2025-10-02", 7.40),
    ("2025-10-03", 7.37),
    ("2025-10-06", 7.29),
    ("2025-10-08", 7.28),
    ("2025-10-09", 7.29),
    ("2025-10-10", 7.34),
    ("2025-10-13", 7.32),
    ("2025-10-14", 7.49),
    ("2025-10-15", 7.49),
    ("2025-10-16", 7.62),
    ("2025-10-17", 7.62),
    ("2025-10-20", 7.75),
    ("2025-10-21", 7.82),
    ("2025-10-22", 7.81),
    ("2025-10-23", 7.88),
    ("2025-10-24", 7.89),
    ("2025-10-27", 7.88),
    ("2025-10-28", 7.90),
    ("2025-10-30", 7.86),
    ("2025-10-31", 7.70),
    ("2025-11-03", 7.94),
    ("2025-11-04", 8.05),
    ("2025-11-05", 8.02),
    ("2025-11-06", 8.14),
    ("2025-11-07", 8.13),
    ("2025-11-10", 8.29),
    ("2025-11-11", 8.33),
    ("2025-11-12", 8.40),
    ("2025-11-13", 8.42),
    ("2025-11-14", 8.35),
    ("2025-11-17", 8.24),
    ("2025-11-18", 8.15),
    ("2025-11-19", 8.13),
    ("2025-11-20", 8.22),
    ("2025-11-21", 8.09),
    ("2025-11-24", 8.21),
    ("2025-11-25", 8.22),
    ("2025-11-26", 8.21),
    ("2025-11-27", 8.23),
    ("2025-11-28", 8.17),
    ("2025-12-01", 8.15),
    ("2025-12-02", 8.16),
    ("2025-12-03", 7.84),
    ("2025-12-04", 7.92),
    ("2025-12-05", 7.98),
    ("2025-12-08", 7.66),
    ("2025-12-09", 7.61),
    ("2025-12-10", 7.57),
    ("2025-12-11", 7.58),
    ("2025-12-12", 7.64),
    ("2025-12-15", 7.55),
    ("2025-12-16", 7.39),
    ("2025-12-17", 7.43),
    ("2025-12-18", 7.52),
    ("2025-12-19", 7.49),
    ("2025-12-22", 7.54),
    ("2025-12-23", 7.61),
    ("2025-12-24", 7.56),
    ("2025-12-29", 7.62),
    ("2025-12-30", 7.72),
]

from datetime import datetime

dates = [datetime.strptime(d, "%Y-%m-%d") for d, _ in price_data]
prices = [p for _, p in price_data]
n_days = len(dates)

maturity_date = datetime.strptime("2025-12-30", "%Y-%m-%d")
start_date = datetime.strptime("2025-06-30", "%Y-%m-%d")

# ---------- Q3(i): Delta Hedging ----------
print(f"\nQ3(i) Delta Hedging Strategy")
print(f"  Short {M} call options on S1 (CCB 939)")
print(f"  Strike K = {K_option}, sigma = {sigma1}, r = {r}")
print(f"  Period: {start_date.strftime('%Y-%m-%d')} to {maturity_date.strftime('%Y-%m-%d')}")
print()

hedging_table = []

for i in range(n_days):
    date = dates[i]
    price = prices[i]
    T_remain = (maturity_date - date).days / 365.0
    num_days_from_prev = (dates[i] - dates[i - 1]).days if i > 0 else 0

    if i < n_days - 1:
        if T_remain > 0:
            d1_val = (np.log(price / K_option) + (r + 0.5 * sigma1**2) * T_remain) / (sigma1 * np.sqrt(T_remain))
            Nd1 = norm.cdf(d1_val)
        else:
            d1_val = float('nan')
            Nd1 = 1.0 if price > K_option else 0.0
        
        delta_shares = Nd1 * M
    else:
        d1_val = float('nan')
        if price > K_option:
            Nd1 = 1.0
            delta_shares = M
        else:
            Nd1 = 0.0
            delta_shares = 0

    if i == 0:
        shares_traded = delta_shares
        cost_of_trade = shares_traded * price
        interest = 0.0
        account_balance = -cost_of_trade
    else:
        prev_balance = hedging_table[i - 1]['account_balance']
        prev_delta_shares = hedging_table[i - 1]['delta_shares']
        
        interest = prev_balance * (np.exp(r * num_days_from_prev / 365.0) - 1)
        
        shares_traded = delta_shares - prev_delta_shares
        cost_of_trade = shares_traded * price
        
        account_balance = prev_balance + interest - cost_of_trade

    hedging_table.append({
        'date': date,
        'T_remain': T_remain,
        'price': price,
        'd1': d1_val,
        'Nd1': Nd1,
        'delta_shares': delta_shares,
        'shares_traded': shares_traded,
        'cost_of_trade': cost_of_trade,
        'interest': interest,
        'account_balance': account_balance,
        'num_days': num_days_from_prev,
    })

final_price = prices[-1]
final_entry = hedging_table[-1]

print(f"Final stock price on maturity: {final_price}")
print(f"Strike price: {K_option}")

if final_price > K_option:
    print(f"Option is IN-THE-MONEY (S_T = {final_price} > K = {K_option})")
    print(f"Option is exercised: sell {M} shares at strike price K = {K_option}")
    exercise_cash = M * K_option
    Fi = final_entry['account_balance'] + exercise_cash
    print(f"  Cash received from exercise = {M} x {K_option} = {exercise_cash:,.2f}")
else:
    print(f"Option is OUT-OF-THE-MONEY (S_T = {final_price} <= K = {K_option})")
    print(f"Option is NOT exercised")
    remaining_shares = final_entry['delta_shares']
    sell_cash = remaining_shares * final_price
    Fi = final_entry['account_balance'] + sell_cash
    print(f"  Sell remaining {remaining_shares:.0f} shares at market price {final_price}")

print(f"\n  Final account balance (before exercise settlement): {final_entry['account_balance']:,.2f}")
print(f"  Fi (final account balance after settlement) = {Fi:,.2f}")

# Print hedging table (selected rows)
print("\n" + "-" * 130)
print(f"{'Date':<12} {'T_rem':>7} {'Price':>7} {'d1':>8} {'N(d1)':>8} {'DeltaShrs':>12} "
      f"{'ShrTraded':>12} {'CostTrade':>14} {'Interest':>12} {'AcctBal':>16}")
print("-" * 130)

# Print first 10, some middle, last 5 rows
rows_to_print = list(range(min(10, n_days)))
mid = n_days // 2
rows_to_print += list(range(max(mid - 2, 10), min(mid + 3, n_days - 5)))
rows_to_print += list(range(max(n_days - 5, 0), n_days))
rows_to_print = sorted(set(rows_to_print))

prev_idx = -1
for idx in rows_to_print:
    if prev_idx >= 0 and idx > prev_idx + 1:
        print(f"{'...':>12}")
    entry = hedging_table[idx]
    d1_str = f"{entry['d1']:8.4f}" if not np.isnan(entry['d1']) else "     N/A"
    print(f"{entry['date'].strftime('%Y-%m-%d'):<12} {entry['T_remain']:7.4f} {entry['price']:7.2f} "
          f"{d1_str} {entry['Nd1']:8.4f} {entry['delta_shares']:12.2f} "
          f"{entry['shares_traded']:12.2f} {entry['cost_of_trade']:14.2f} "
          f"{entry['interest']:12.2f} {entry['account_balance']:16.2f}")
    prev_idx = idx

print("-" * 130)

# ---------- Q3(ii): Option Premium Deposit ----------
print(f"\nQ3(ii) Option Premium Deposit")

T3 = (maturity_date - start_date).days / 365.0
print(f"  T (June 30 to Dec 30) = {(maturity_date - start_date).days} days = {T3:.6f} years")

d1_q3 = (np.log(S1_0 / K_option) + (r + 0.5 * sigma1**2) * T3) / (sigma1 * np.sqrt(T3))
d2_q3 = d1_q3 - sigma1 * np.sqrt(T3)
bs_price_q3 = S1_0 * norm.cdf(d1_q3) - K_option * np.exp(-r * T3) * norm.cdf(d2_q3)

print(f"  BS call price (T={T3:.6f}) = {bs_price_q3:.6f}")

premium_received = bs_price_q3 * M
print(f"  Total premium received = {bs_price_q3:.6f} x {M} = {premium_received:,.2f}")

Pi = premium_received * np.exp(r * T3)
print(f"  Pi (premium with interest to maturity) = {premium_received:,.2f} x exp({r} x {T3:.6f})")
print(f"  Pi = {Pi:,.2f}")

print(f"\n  Comparison:")
print(f"    Fi (hedging final balance) = {Fi:,.2f}")
print(f"    Pi (deposit maturity value) = {Pi:,.2f}")
print(f"    |Fi| - Pi = {abs(Fi) - Pi:,.2f}")
print(f"    Ratio |Fi|/Pi = {abs(Fi)/Pi:.6f}")

print(f"""
  Comment:
  The final hedging account balance Fi ({Fi:,.2f}) is negative, while the 
  deposit amount Pi ({Pi:,.2f}) is positive. Under the Black-Scholes framework,
  if delta hedging is performed continuously, |Fi| should exactly equal Pi, 
  meaning the cost of hedging equals the option premium received. In practice,
  since we hedge only at discrete daily intervals, there is a small discrepancy 
  between |Fi| and Pi. This difference arises from discrete hedging error 
  (gamma risk) and the fact that realized volatility may differ from the 
  implied volatility used. The closer |Fi| is to Pi, the more effective the 
  delta hedging strategy is.
""")

print("=" * 80)
print("END OF ASSIGNMENT")
print("=" * 80)
