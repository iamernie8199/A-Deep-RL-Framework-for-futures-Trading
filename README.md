# A-Deep-RL-Framework-for-Index-futures-Trading

[![hackmd-github-sync-badge](https://hackmd.io/xuu0j4MlSZqIQR_hB9VxhQ/badge)](https://hackmd.io/xuu0j4MlSZqIQR_hB9VxhQ)

## Data
- 台指期
    - 交易時間: 前日夜盤+今日日盤

## feature
- log return
    - $\ln\frac{P_t}{P_{t-1}}$, $P_t$為資產在t時刻的價格
- 基差(basis)
    - def: 現貨價格 - 期貨價格
    - $\frac{現貨-期貨}{現貨}\times10$
        - 乘10目的是放大原始值域([-0.03, 0.07]->[-0.3, 0.7])
- 成交量
    - percentile: data[:t], 統計t時刻值在t時刻前所有資料的百分位數
        - 刻劃破新高
    - deg_change: $\arctan(V_t-V_{t-1}, 100)$
        - 刻劃變化量, 100為歷史成交量下限

## ref
### fractal
- the first 100 pages of the book,"The Science of Fractal Images" edited by Heinz-Otto Peitgen and Dietmar Saupe

### FinRL
https://towardsdatascience.com/deep-reinforcement-learning-for-automated-stock-trading-f1dad0126a02

## todo
- use [Dask df](https://examples.dask.org/dataframe.html) to improve speed