 #  A Whale off the Port(folio)
 ---

In this assignment, you'll get to use what you've learned this week to evaluate the performance among various algorithmic, hedge, and mutual fund portfolios and compare them against the S&P 500 Index.


```python
# Initial imports
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

%matplotlib inline
```

# Data Cleaning

In this section, you will need to read the CSV files into DataFrames and perform any necessary data cleaning steps. After cleaning, combine all DataFrames into a single DataFrame.

Files:

* `whale_returns.csv`: Contains returns of some famous "whale" investors' portfolios.

* `algo_returns.csv`: Contains returns from the in-house trading algorithms from Harold's company.

* `sp500_history.csv`: Contains historical closing prices of the S&P 500 Index.


```python
# define all paths for my input data sets
aapl_path = Path('Resources/aapl_historical.csv')
algo_path = Path('Resources/algo_returns.csv')
cost_historical_path = Path('Resources/cost_historical.csv')
goog_historical_path = Path('Resources/goog_historical.csv')
sp500_path = Path('Resources/sp500_history.csv')
whale_returns_path = Path('Resources/whale_returns.csv')
```

## Whale Returns

Read the Whale Portfolio daily returns and clean the data


```python
# Reading whale returns
whale_returns = pd.read_csv(whale_returns_path, index_col='Date', parse_dates=True)
whale_returns.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SOROS FUND MANAGEMENT LLC</th>
      <th>PAULSON &amp; CO.INC.</th>
      <th>TIGER GLOBAL MANAGEMENT LLC</th>
      <th>BERKSHIRE HATHAWAY INC</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-03-02</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-03-03</th>
      <td>-0.001266</td>
      <td>-0.004981</td>
      <td>-0.000496</td>
      <td>-0.006569</td>
    </tr>
    <tr>
      <th>2015-03-04</th>
      <td>0.002230</td>
      <td>0.003241</td>
      <td>-0.002534</td>
      <td>0.004213</td>
    </tr>
    <tr>
      <th>2015-03-05</th>
      <td>0.004016</td>
      <td>0.004076</td>
      <td>0.002355</td>
      <td>0.006726</td>
    </tr>
    <tr>
      <th>2015-03-06</th>
      <td>-0.007905</td>
      <td>-0.003574</td>
      <td>-0.008481</td>
      <td>-0.013098</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Count nulls
whale_returns.isnull().sum()
```




    SOROS FUND MANAGEMENT LLC      1
    PAULSON & CO.INC.              1
    TIGER GLOBAL MANAGEMENT LLC    1
    BERKSHIRE HATHAWAY INC         1
    dtype: int64




```python
# Drop nulls
whale_returns.dropna(inplace=True)
whale_returns.isnull().sum()
```




    SOROS FUND MANAGEMENT LLC      0
    PAULSON & CO.INC.              0
    TIGER GLOBAL MANAGEMENT LLC    0
    BERKSHIRE HATHAWAY INC         0
    dtype: int64



## Algorithmic Daily Returns

Read the algorithmic daily returns and clean the data


```python
# Reading algorithmic returns
algo_retuns = pd.read_csv(algo_path, index_col='Date', parse_dates=True)
algo_retuns.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Algo 1</th>
      <th>Algo 2</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-05-28</th>
      <td>0.001745</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-05-29</th>
      <td>0.003978</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-05-30</th>
      <td>0.004464</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-06-02</th>
      <td>0.005692</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2014-06-03</th>
      <td>0.005292</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Count nulls
algo_retuns.isnull().sum()
```




    Algo 1    0
    Algo 2    6
    dtype: int64




```python
# Drop nulls
algo_retuns.dropna(inplace=True)
algo_retuns.isnull().sum()
```




    Algo 1    0
    Algo 2    0
    dtype: int64



## S&P 500 Returns

Read the S&P 500 historic closing prices and create a new daily returns DataFrame from the data. 


```python
# Reading S&P 500 Closing Prices
sp500 = pd.read_csv(sp500_path, index_col='Date', parse_dates=True)
sp500.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-04-23</th>
      <td>$2933.68</td>
    </tr>
    <tr>
      <th>2019-04-22</th>
      <td>$2907.97</td>
    </tr>
    <tr>
      <th>2019-04-18</th>
      <td>$2905.03</td>
    </tr>
    <tr>
      <th>2019-04-17</th>
      <td>$2900.45</td>
    </tr>
    <tr>
      <th>2019-04-16</th>
      <td>$2907.06</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check Data Types
sp500.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 1649 entries, 2019-04-23 to 2012-10-01
    Data columns (total 1 columns):
     #   Column  Non-Null Count  Dtype 
    ---  ------  --------------  ----- 
     0   Close   1649 non-null   object
    dtypes: object(1)
    memory usage: 25.8+ KB
    


```python
# Fix Data Types
sp500['Close'] = sp500['Close'].replace({'\$': '', ',': ''}, regex=True).astype(float)
sp500.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-04-23</th>
      <td>2933.68</td>
    </tr>
    <tr>
      <th>2019-04-22</th>
      <td>2907.97</td>
    </tr>
    <tr>
      <th>2019-04-18</th>
      <td>2905.03</td>
    </tr>
    <tr>
      <th>2019-04-17</th>
      <td>2900.45</td>
    </tr>
    <tr>
      <th>2019-04-16</th>
      <td>2907.06</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Calculate Daily Returns
sp500['Close'] = sp500['Close'].pct_change()
sp500.dropna(inplace=True)
sp500.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-04-22</th>
      <td>-0.008764</td>
    </tr>
    <tr>
      <th>2019-04-18</th>
      <td>-0.001011</td>
    </tr>
    <tr>
      <th>2019-04-17</th>
      <td>-0.001577</td>
    </tr>
    <tr>
      <th>2019-04-16</th>
      <td>0.002279</td>
    </tr>
    <tr>
      <th>2019-04-15</th>
      <td>-0.000509</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Drop nulls
sp500.isnull().sum()
```




    Close    0
    dtype: int64




```python
# Rename `Close` Column to be specific to this portfolio.
sp500.columns = ["SP500"]
sp500.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SP500</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-04-22</th>
      <td>-0.008764</td>
    </tr>
    <tr>
      <th>2019-04-18</th>
      <td>-0.001011</td>
    </tr>
    <tr>
      <th>2019-04-17</th>
      <td>-0.001577</td>
    </tr>
    <tr>
      <th>2019-04-16</th>
      <td>0.002279</td>
    </tr>
    <tr>
      <th>2019-04-15</th>
      <td>-0.000509</td>
    </tr>
  </tbody>
</table>
</div>



## Combine Whale, Algorithmic, and S&P 500 Returns


```python
# Join Whale Returns, Algorithmic Returns, and the S&P 500 Returns into a single DataFrame with columns for each portfolio's returns.
daily_returns = pd.concat([whale_returns,algo_retuns,sp500], axis="columns", join="inner")
daily_returns.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SOROS FUND MANAGEMENT LLC</th>
      <th>PAULSON &amp; CO.INC.</th>
      <th>TIGER GLOBAL MANAGEMENT LLC</th>
      <th>BERKSHIRE HATHAWAY INC</th>
      <th>Algo 1</th>
      <th>Algo 2</th>
      <th>SP500</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-03-03</th>
      <td>-0.001266</td>
      <td>-0.004981</td>
      <td>-0.000496</td>
      <td>-0.006569</td>
      <td>-0.001942</td>
      <td>-0.000949</td>
      <td>0.004408</td>
    </tr>
    <tr>
      <th>2015-03-04</th>
      <td>0.002230</td>
      <td>0.003241</td>
      <td>-0.002534</td>
      <td>0.004213</td>
      <td>-0.008589</td>
      <td>0.002416</td>
      <td>-0.001195</td>
    </tr>
    <tr>
      <th>2015-03-05</th>
      <td>0.004016</td>
      <td>0.004076</td>
      <td>0.002355</td>
      <td>0.006726</td>
      <td>-0.000955</td>
      <td>0.004323</td>
      <td>0.014378</td>
    </tr>
    <tr>
      <th>2015-03-06</th>
      <td>-0.007905</td>
      <td>-0.003574</td>
      <td>-0.008481</td>
      <td>-0.013098</td>
      <td>-0.004957</td>
      <td>-0.011460</td>
      <td>-0.003929</td>
    </tr>
    <tr>
      <th>2015-03-09</th>
      <td>0.000582</td>
      <td>0.004225</td>
      <td>0.005843</td>
      <td>-0.001652</td>
      <td>-0.005447</td>
      <td>0.001303</td>
      <td>0.017254</td>
    </tr>
  </tbody>
</table>
</div>



---

# Conduct Quantitative Analysis

In this section, you will calculate and visualize performance and risk metrics for the portfolios.

## Performance Anlysis

#### Calculate and Plot the daily returns.


```python
# Plot daily returns of all portfolios
daily_returns.plot(figsize=(25,10))
```




    <AxesSubplot:xlabel='Date'>




    
![png](Resources/Images/output_24_1.png)
    


#### Calculate and Plot cumulative returns.


```python
# Calculate cumulative returns of all portfolios
cumulative_returns = (1+daily_returns).cumprod()
# Plot cumulative returns
cumulative_returns.plot(figsize=(25,10))
```




    <AxesSubplot:xlabel='Date'>




    
![png](Resources/Images/output_26_1.png)
    


---

## Risk Analysis

Determine the _risk_ of each portfolio:

1. Create a box plot for each portfolio. 
2. Calculate the standard deviation for all portfolios
4. Determine which portfolios are riskier than the S&P 500
5. Calculate the Annualized Standard Deviation

### Create a box plot for each portfolio



```python
# Box plot to visually show risk
daily_returns.plot.box(figsize=(25,10))
```




    <AxesSubplot:>




    
![png](Resources/Images/output_30_1.png)
    


### Calculate Standard Deviations


```python
# Calculate the daily standard deviations of all portfolios
daily_std = daily_returns.std()
daily_std
```




    SOROS FUND MANAGEMENT LLC      0.007896
    PAULSON & CO.INC.              0.007026
    TIGER GLOBAL MANAGEMENT LLC    0.010897
    BERKSHIRE HATHAWAY INC         0.012919
    Algo 1                         0.007623
    Algo 2                         0.008341
    SP500                          0.008587
    dtype: float64



### Determine which portfolios are riskier than the S&P 500


```python
# Calculate  the daily standard deviation of S&P 500
    # the daily std of S&P 500 is already computed in the daily_std dataframe
daily_std['SP500']
# Determine which portfolios are riskier than the S&P 500
    # Checking out what portfolios have a greater STD than the STD if SP500
high_risk_portfolios=daily_std[daily_std>daily_std['SP500']]
#alternative, but less code efficient
    #high_risk_portfolios = daily_std.where(daily_std>daily_std['SP500'])
    #high_risk_portfolios.dropna(inplace=True)
high_risk_portfolios
```




    TIGER GLOBAL MANAGEMENT LLC    0.010897
    BERKSHIRE HATHAWAY INC         0.012919
    dtype: float64



### Calculate the Annualized Standard Deviation


```python
# Calculate the annualized standard deviation (252 trading days)
annualized_std = daily_std * np.sqrt(252)
annualized_std
```




    SOROS FUND MANAGEMENT LLC      0.125348
    PAULSON & CO.INC.              0.111527
    TIGER GLOBAL MANAGEMENT LLC    0.172989
    BERKSHIRE HATHAWAY INC         0.205079
    Algo 1                         0.121006
    Algo 2                         0.132413
    SP500                          0.136313
    dtype: float64



---

## Rolling Statistics

Risk changes over time. Analyze the rolling statistics for Risk and Beta. 

1. Calculate and plot the rolling standard deviation for the S&P 500 using a 21-day window
2. Calculate the correlation between each stock to determine which portfolios may mimick the S&P 500
3. Choose one portfolio, then calculate and plot the 60-day rolling beta between it and the S&P 500

### Calculate and plot rolling `std` for all portfolios with 21-day window


```python
# Calculate the rolling standard deviation for all portfolios using a 21-day window
daily_returns['SP500'].rolling(window=21).std().plot(figsize=(25,10))
# Plot the rolling standard deviation

```




    <AxesSubplot:xlabel='Date'>




    
![png](Resources/Images/output_40_1.png)
    


### Calculate and plot the correlation


```python
# Calculate the correlation
correlation = daily_returns.corr()
correlation
# Display de correlation matrix
plt.rcParams['figure.figsize'] = (25.0, 10.0)
plt.rcParams['font.family'] = "consolas"
plt.rcParams.update({'font.size': 20})

sns.heatmap(correlation, annot=True, vmin=-1, vmax=1, annot_kws={'size':25})
```




    <AxesSubplot:>




    
![png](Resources/Images/output_42_1.png)
    


### Calculate and Plot Beta for a chosen portfolio and the S&P 500


```python
# Calculate covariance of a single portfolio
covariance = daily_returns['SOROS FUND MANAGEMENT LLC'].cov(daily_returns['SP500'])
# Calculate variance of S&P 500
variance = daily_returns['SP500'].var()
# Computing beta
beta = covariance / variance
# Plot beta trend
rolling_covariance = daily_returns['SOROS FUND MANAGEMENT LLC'].rolling(window=60).cov(daily_returns['SP500'])
rolling_covariance.plot(figsize=(25, 10), title='Rolling 60-Day Covariance of SOROS FUND MANAGEMENT LLC Returns vs. S&P 500 Returns')
```




    <AxesSubplot:title={'center':'Rolling 60-Day Covariance of SOROS FUND MANAGEMENT LLC Returns vs. S&P 500 Returns'}, xlabel='Date'>




    
![png](Resources/Images/output_44_1.png)
    


## Rolling Statistics Challenge: Exponentially Weighted Average 

An alternative way to calculate a rolling window is to take the exponentially weighted moving average. This is like a moving window average, but it assigns greater importance to more recent observations. Try calculating the [`ewm`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html) with a 21-day half-life.


```python
# Use `ewm` to calculate the rolling window
rolling_window_ewm = daily_returns.ewm(halflife="21 days", times=daily_returns.index).mean()
rolling_window_ewm.plot()
```




    <AxesSubplot:xlabel='Date'>




    
![png](Resources/Images/output_46_1.png)
    


---

# Sharpe Ratios
In reality, investment managers and thier institutional investors look at the ratio of return-to-risk, and not just returns alone. After all, if you could invest in one of two portfolios, and each offered the same 10% return, yet one offered lower risk, you'd take that one, right?

### Using the daily returns, calculate and visualize the Sharpe ratios using a bar plot


```python
# Annualized Sharpe Ratios
annualized_sharpe_ratios = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
annualized_sharpe_ratios.sort_values()
```




    SP500                         -0.518582
    PAULSON & CO.INC.             -0.491422
    TIGER GLOBAL MANAGEMENT LLC   -0.130186
    SOROS FUND MANAGEMENT LLC      0.342894
    Algo 2                         0.484334
    BERKSHIRE HATHAWAY INC         0.606743
    Algo 1                         1.369589
    dtype: float64




```python
# Visualize the sharpe ratios as a bar plot
annualized_sharpe_ratios.plot.bar()
```




    <AxesSubplot:>




    
![png](Resources/Images/output_50_1.png)
    


### Determine whether the algorithmic strategies outperform both the market (S&P 500) and the whales portfolios.

Algo 1 outperformed market and whales while algo 2 fell below market.

---

# Create Custom Portfolio

In this section, you will build your own portfolio of stocks, calculate the returns, and compare the results to the Whale Portfolios and the S&P 500. 

1. Choose 3-5 custom stocks with at last 1 year's worth of historic prices and create a DataFrame of the closing prices and dates for each stock.
2. Calculate the weighted returns for the portfolio assuming an equal number of shares for each stock
3. Join your portfolio returns to the DataFrame that contains all of the portfolio returns
4. Re-run the performance and risk analysis with your portfolio to see how it compares to the others
5. Include correlation analysis to determine which stocks (if any) are correlated

## Choose 3-5 custom stocks with at last 1 year's worth of historic prices and create a DataFrame of the closing prices and dates for each stock.

For this demo solution, we fetch data from three companies listes in the S&P 500 index.

* `GOOG` - [Google, LLC](https://en.wikipedia.org/wiki/Google)

* `AAPL` - [Apple Inc.](https://en.wikipedia.org/wiki/Apple_Inc.)

* `COST` - [Costco Wholesale Corporation](https://en.wikipedia.org/wiki/Costco)


```python
# Reading data from 1st stock
goog = pd.read_csv(goog_historical_path, index_col="Trade DATE", infer_datetime_format=True, parse_dates=True).iloc[:, 1:]
goog.columns = ['goog']
goog.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>goog</th>
    </tr>
    <tr>
      <th>Trade DATE</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-05-09</th>
      <td>1162.38</td>
    </tr>
    <tr>
      <th>2019-05-08</th>
      <td>1166.27</td>
    </tr>
    <tr>
      <th>2019-05-07</th>
      <td>1174.10</td>
    </tr>
    <tr>
      <th>2019-05-06</th>
      <td>1189.39</td>
    </tr>
    <tr>
      <th>2019-05-03</th>
      <td>1185.40</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Reading data from 2nd stock
aapl = pd.read_csv(aapl_path, index_col="Trade DATE", infer_datetime_format=True, parse_dates=True).iloc[:, 1:]
aapl.columns = ['aapl']
aapl.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>aapl</th>
    </tr>
    <tr>
      <th>Trade DATE</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-05-09</th>
      <td>200.72</td>
    </tr>
    <tr>
      <th>2019-05-08</th>
      <td>202.90</td>
    </tr>
    <tr>
      <th>2019-05-07</th>
      <td>202.86</td>
    </tr>
    <tr>
      <th>2019-05-06</th>
      <td>208.48</td>
    </tr>
    <tr>
      <th>2019-05-03</th>
      <td>211.75</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Reading data from 3rd stock
cost = pd.read_csv(cost_historical_path, index_col="Trade DATE", infer_datetime_format=True, parse_dates=True).iloc[:, 1:]
cost.columns = ['cost']
cost.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cost</th>
    </tr>
    <tr>
      <th>Trade DATE</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-05-09</th>
      <td>243.47</td>
    </tr>
    <tr>
      <th>2019-05-08</th>
      <td>241.34</td>
    </tr>
    <tr>
      <th>2019-05-07</th>
      <td>240.18</td>
    </tr>
    <tr>
      <th>2019-05-06</th>
      <td>244.23</td>
    </tr>
    <tr>
      <th>2019-05-03</th>
      <td>244.62</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Combine all stocks in a single DataFrame
combined_stocks = pd.concat([goog,aapl,cost], axis='columns', join='inner')
combined_stocks.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>goog</th>
      <th>aapl</th>
      <th>cost</th>
    </tr>
    <tr>
      <th>Trade DATE</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-05-09</th>
      <td>1162.38</td>
      <td>200.72</td>
      <td>243.47</td>
    </tr>
    <tr>
      <th>2019-05-08</th>
      <td>1166.27</td>
      <td>202.90</td>
      <td>241.34</td>
    </tr>
    <tr>
      <th>2019-05-07</th>
      <td>1174.10</td>
      <td>202.86</td>
      <td>240.18</td>
    </tr>
    <tr>
      <th>2019-05-06</th>
      <td>1189.39</td>
      <td>208.48</td>
      <td>244.23</td>
    </tr>
    <tr>
      <th>2019-05-03</th>
      <td>1185.40</td>
      <td>211.75</td>
      <td>244.62</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Reset Date index
combined_stocks.index.names = ["Date"]
combined_stocks.sort_index(inplace=True)
```


```python
# Reorganize portfolio data by having a column per symbol
# already managed this few steps before
```


```python
# Calculate daily returns
combined_stocks = combined_stocks.pct_change()
# Drop NAs
combined_stocks.dropna(inplace=True)
# Display sample data
combined_stocks.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>goog</th>
      <th>aapl</th>
      <th>cost</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-05-14</th>
      <td>0.001766</td>
      <td>-0.002333</td>
      <td>0.000613</td>
    </tr>
    <tr>
      <th>2018-05-15</th>
      <td>-0.019060</td>
      <td>-0.009088</td>
      <td>-0.002042</td>
    </tr>
    <tr>
      <th>2018-05-16</th>
      <td>0.002354</td>
      <td>0.009333</td>
      <td>0.016523</td>
    </tr>
    <tr>
      <th>2018-05-17</th>
      <td>-0.002940</td>
      <td>-0.006324</td>
      <td>0.004479</td>
    </tr>
    <tr>
      <th>2018-05-18</th>
      <td>-0.011339</td>
      <td>-0.003637</td>
      <td>-0.003206</td>
    </tr>
  </tbody>
</table>
</div>



## Calculate the weighted returns for the portfolio assuming an equal number of shares for each stock


```python
# Set weights
weights = [1/3, 1/3, 1/3]

# Calculate portfolio return
weighted_returns = combined_stocks.dot(weights)
# Display sample data
weighted_returns.head()
```




    Date
    2018-05-14    0.000015
    2018-05-15   -0.010064
    2018-05-16    0.009403
    2018-05-17   -0.001595
    2018-05-18   -0.006061
    dtype: float64



## Join your portfolio returns to the DataFrame that contains all of the portfolio returns


```python
# Join your returns DataFrame to the original returns DataFrame
weighted_returns = pd.concat([daily_returns,weighted_returns], axis="columns", join="inner")
weighted_returns.rename(columns = {0:"my_portfolio"}, inplace = True)
weighted_returns.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SOROS FUND MANAGEMENT LLC</th>
      <th>PAULSON &amp; CO.INC.</th>
      <th>TIGER GLOBAL MANAGEMENT LLC</th>
      <th>BERKSHIRE HATHAWAY INC</th>
      <th>Algo 1</th>
      <th>Algo 2</th>
      <th>SP500</th>
      <th>my_portfolio</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-05-14</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000915</td>
      <td>0.001635</td>
      <td>0.006889</td>
      <td>0.000015</td>
    </tr>
    <tr>
      <th>2018-05-15</th>
      <td>-0.000726</td>
      <td>-0.001409</td>
      <td>-0.003189</td>
      <td>-0.014606</td>
      <td>-0.001135</td>
      <td>-0.001139</td>
      <td>-0.004044</td>
      <td>-0.010064</td>
    </tr>
    <tr>
      <th>2018-05-16</th>
      <td>0.008637</td>
      <td>0.006244</td>
      <td>0.005480</td>
      <td>0.004310</td>
      <td>-0.002326</td>
      <td>0.003341</td>
      <td>0.000857</td>
      <td>0.009403</td>
    </tr>
    <tr>
      <th>2018-05-17</th>
      <td>-0.001955</td>
      <td>0.002524</td>
      <td>-0.006267</td>
      <td>-0.005140</td>
      <td>-0.006949</td>
      <td>0.005205</td>
      <td>0.002639</td>
      <td>-0.001595</td>
    </tr>
    <tr>
      <th>2018-05-18</th>
      <td>-0.004357</td>
      <td>-0.002672</td>
      <td>-0.012832</td>
      <td>-0.002212</td>
      <td>0.002557</td>
      <td>-0.002496</td>
      <td>-0.007333</td>
      <td>-0.006061</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Only compare dates where return data exists for all the stocks (drop NaNs)
weighted_returns.dropna()
weighted_returns.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SOROS FUND MANAGEMENT LLC</th>
      <th>PAULSON &amp; CO.INC.</th>
      <th>TIGER GLOBAL MANAGEMENT LLC</th>
      <th>BERKSHIRE HATHAWAY INC</th>
      <th>Algo 1</th>
      <th>Algo 2</th>
      <th>SP500</th>
      <th>my_portfolio</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-05-14</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000915</td>
      <td>0.001635</td>
      <td>0.006889</td>
      <td>0.000015</td>
    </tr>
    <tr>
      <th>2018-05-15</th>
      <td>-0.000726</td>
      <td>-0.001409</td>
      <td>-0.003189</td>
      <td>-0.014606</td>
      <td>-0.001135</td>
      <td>-0.001139</td>
      <td>-0.004044</td>
      <td>-0.010064</td>
    </tr>
    <tr>
      <th>2018-05-16</th>
      <td>0.008637</td>
      <td>0.006244</td>
      <td>0.005480</td>
      <td>0.004310</td>
      <td>-0.002326</td>
      <td>0.003341</td>
      <td>0.000857</td>
      <td>0.009403</td>
    </tr>
    <tr>
      <th>2018-05-17</th>
      <td>-0.001955</td>
      <td>0.002524</td>
      <td>-0.006267</td>
      <td>-0.005140</td>
      <td>-0.006949</td>
      <td>0.005205</td>
      <td>0.002639</td>
      <td>-0.001595</td>
    </tr>
    <tr>
      <th>2018-05-18</th>
      <td>-0.004357</td>
      <td>-0.002672</td>
      <td>-0.012832</td>
      <td>-0.002212</td>
      <td>0.002557</td>
      <td>-0.002496</td>
      <td>-0.007333</td>
      <td>-0.006061</td>
    </tr>
  </tbody>
</table>
</div>



## Re-run the risk analysis with your portfolio to see how it compares to the others

### Calculate the Annualized Standard Deviation


```python
# Calculate the annualized `std`
annualized_std_weighted_returns = weighted_returns.std()* np.sqrt(252)
annualized_std_weighted_returns
```




    SOROS FUND MANAGEMENT LLC      0.146812
    PAULSON & CO.INC.              0.116928
    TIGER GLOBAL MANAGEMENT LLC    0.232898
    BERKSHIRE HATHAWAY INC         0.247305
    Algo 1                         0.133927
    Algo 2                         0.139499
    SP500                          0.152469
    my_portfolio                   0.211627
    dtype: float64



### Calculate and plot rolling `std` with 21-day window


```python
# Calculate rolling standard deviation
weighted_returns.rolling(window=21).std().plot(figsize=(25,10))
# Plot rolling standard deviation

```




    <AxesSubplot:xlabel='Date'>




    
![png](Resources/Images/output_71_1.png)
    


### Calculate and plot the correlation


```python
# Calculate and plot the correlation
correlation2 = weighted_returns.corr()
plt.rcParams['figure.figsize'] = (25.0, 10.0)
plt.rcParams['font.family'] = "consolas"
plt.rcParams.update({'font.size': 20})

sns.heatmap(correlation2, annot=True, vmin=-1, vmax=1, annot_kws={'size':25})
```




    <AxesSubplot:>




    
![png](Resources/Images/output_73_1.png)
    


### Calculate and Plot Rolling 60-day Beta for Your Portfolio compared to the S&P 500


```python
# Calculate and plot Beta
rolling_covariance = weighted_returns['my_portfolio'].rolling(window=60).cov(weighted_returns['SP500'])
rolling_variance = weighted_returns['SP500'].rolling(window=60).var()
rolling_beta = rolling_covariance / rolling_variance

rolling_variance.plot(figsize=(25, 10), title='Rolling 60-Day Beta of my portfolio vs. S&P 500 Returns')
```




    <AxesSubplot:title={'center':'Rolling 60-Day Beta of my portfolio vs. S&P 500 Returns'}, xlabel='Date'>




    
![png](Resources/Images/output_75_1.png)
    


### Using the daily returns, calculate and visualize the Sharpe ratios using a bar plot


```python
# Calculate Annualzied Sharpe Ratios
annualized_sharpe_ratios2 = (weighted_returns.mean() * 252) / (weighted_returns.std() * np.sqrt(252))
annualized_sharpe_ratios2.sort_values()
annualized_sharpe_ratios2
```




    SOROS FUND MANAGEMENT LLC      0.380007
    PAULSON & CO.INC.              0.227577
    TIGER GLOBAL MANAGEMENT LLC   -1.066635
    BERKSHIRE HATHAWAY INC         0.103006
    Algo 1                         2.001260
    Algo 2                         0.007334
    SP500                         -0.427676
    my_portfolio                   0.876152
    dtype: float64




```python
# Visualize the sharpe ratios as a bar plot
annualized_sharpe_ratios2.plot.bar()
```




    <AxesSubplot:>




    
![png](Resources/Images/output_78_1.png)
    


### How does your portfolio do?

My portfolio is performing OK, only trailing Algo 1, but beating all other portfolios. My portfolio is riskier than others


```python
annualized_std_weighted_returns.sort_values()
annualized_std_weighted_returns

```




    SOROS FUND MANAGEMENT LLC      0.146812
    PAULSON & CO.INC.              0.116928
    TIGER GLOBAL MANAGEMENT LLC    0.232898
    BERKSHIRE HATHAWAY INC         0.247305
    Algo 1                         0.133927
    Algo 2                         0.139499
    SP500                          0.152469
    my_portfolio                   0.211627
    dtype: float64


