import pandas as pd
import numpy as np
from pathlib import Path

harold_portfolio_path = Path('harold_portfolio.csv')
my_portfolio_path = Path('my_portfolio.csv')

harold_csv = pd.read_csv(harold_portfolio_path, header=0, parse_dates=True, index_col='Date', infer_datetime_format=True)
my_portfolio_csv = pd.read_csv(my_portfolio_path, header=0, parse_dates=True, index_col='Date', infer_datetime_format=True)
harold_csv= harold_csv.sort_index(ascending=True)
my_portfolio_csv= my_portfolio_csv.sort_index(ascending=True)

print(harold_csv.head(5))
print(my_portfolio_csv.head(5))
print('========================')

harold_daily_returns = harold_csv.pct_change().dropna()
my_daily_returns = my_portfolio_csv.pct_change().dropna()

print(harold_daily_returns.head(5))
print(my_daily_returns.head(5))
print('========================')

merged_daily_returns = pd.concat([harold_daily_returns, my_daily_returns], ignore_index=False,
                                 axis='rows', join='inner').drop_duplicates()
print(merged_daily_returns)
print('========================')

st_dev_numpy = np.std(merged_daily_returns)
st_dev2_pandas = merged_daily_returns.std()
print("numpy: \n",st_dev_numpy)
print("pandas: \n",st_dev2_pandas)
print('========================')

sharpe_ratios = (merged_daily_returns.mean() * 252) / (merged_daily_returns.std() * np.sqrt(252))

print(sharpe_ratios)
print('========================')
sharpe_ratios.plot.bar()
