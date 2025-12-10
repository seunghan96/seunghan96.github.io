```
import pandas as pd
import numpy as np

# calculate std & sharpe ratio of portfolio
def get_std_and_sharpe(w_, mean_returns_, cov_matrix_):
    portfolio_return_ = np.dot(w_, mean_returns_)
    portfolio_std_ = np.sqrt(np.dot(w_.T, np.dot(cov_matrix_, w_)))
    if portfolio_std_ !=0:
        sharpe_ = portfolio_return_ / portfolio_std_
    else:
        sharpe_ = 0
    return portfolio_std_, sharpe_

def markowitz_stock_portfolio():
    # don't remove these lines
    np.random.seed(123)
    
    # (1) Load dataset
    df = pd.read_csv("data/stock-prices.csv")
    df.set_index("Date", inplace = True)
    N_assets = len(df.columns)
    N_portfolios = 10000

    # (2) Calculate statistics
    returns = df.pct_change()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    # (3) Calculate return & std of portfolios
    results = []
    for _ in range(N_portfolios):
        w = np.random.random(N_assets)
        w /= np.sum(w)
        portfolio_std, sharpe = get_std_and_sharpe(w, mean_returns, cov_matrix)
        results.append(np.concatenate([w, [portfolio_std, sharpe]]))
    
    # (4) Save results
    portfolios = pd.DataFrame(results)
    portfolios.columns = list(df.columns) + ["std", "sharpe"]

    # (5) Get optimal portfolios
    stock_min_std = portfolios.loc[portfolios["std"].idxmin()].copy()
    stock_max_sharpe = portfolios.loc[portfolios["sharpe"].idxmax()].copy()
    stock_min_std[df.columns] *= 100
    stock_max_sharpe[df.columns] *= 100

    optimal_portfolios = pd.DataFrame([stock_min_std, stock_max_sharpe])
    optimal_portfolios.index = ["min_std", "max_sharpe"]
    
    return {
        "returns": returns,
        "mean_returns": mean_returns,
        "cov_matrix": cov_matrix,
        "optimal_portfolios": optimal_portfolios,
    }
```

