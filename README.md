# A-Deep-RL-Framework-for-Index-futures-Trading

[![hackmd-github-sync-badge](https://hackmd.io/xuu0j4MlSZqIQR_hB9VxhQ/badge)](https://hackmd.io/xuu0j4MlSZqIQR_hB9VxhQ)

## Data
- 台指期
    - Source:
        - DQ2: 1987/1/7 ~ 2021/3/15
        - 期交所: 2021/3/15 ~ 
    - 交易時間: 前日夜盤+今日日盤
    - 其他說明:
        - DQ2成交量: 不含價差交易

## feature
- log return
    - $\ln\frac{P_t}{P_{t-1}}$, $P_t$為資產在t時刻的價格
- 基差(basis)
    - def: 現貨價格 - 期貨價格
    - $\frac{現貨-期貨}{現貨}\times10$
        - 乘10目的是放大原始值域([-0.03, 0.07]->[-0.3, 0.7])
- 成交量/OI
    - percentile: data[:t], 統計t時刻值在t時刻前所有資料的百分位數
        - 刻劃破新高
    - deg_change: $\arctan(V_t-V_{t-1}, 100)$
        - 刻劃變化量, 100為歷史成交量下限
- K棒
    - 實體: $|o-c|$
    - 上影線: $high-\max(o, c)$
    - 下影線: $\min(o,c)-low$
    - 除range(h-l), 以分形角度看K棒
- OHLC
    - [Drift‐Independent Volatility Estimation Based on High, Low, Open, and Close Prices](https://www.jstor.org/stable/10.1086/209650?seq=1#metadata_info_tab_contents)
    - $o_t = \ln(O_t)-\ln(C_{t-1})$
    - $u_t = \ln(H_t)-\ln(O_t)$
    - $d_t = \ln(L_t)-\ln(O_t)$
    - $c_t = \ln(C_t)-\ln(O_t)$
- Hurst exponent
    - [Hurst Exponent and Trading Signals Derived from Market Time Series](https://www.scitepress.org/Papers/2018/66670/66670.pdf)
    - $p(k) = Ck^{-\alpha}$
        - p(k) is an autocorrelation function
        - C is constant
        - k is number of lags
        - $\alpha$ is the decay parameter
            - $\alpha$ ranges between [0,2]
            - $0<\alpha<1$: persistence
            - $\alpha=1$: random walk
            - $1<\alpha<2$: mean reversion
    - $H=1-\frac{\alpha}{2}$
        - H=0.5: random walk
        - H<0.5: mean reversion
        - H>0.5: momentum
    - 2Q/4Q

## ref
### fractal
- the first 100 pages of the book,"The Science of Fractal Images" edited by Heinz-Otto Peitgen and Dietmar Saupe

### FinRL
https://towardsdatascience.com/deep-reinforcement-learning-for-automated-stock-trading-f1dad0126a02

## todo
- use [Dask df](https://examples.dask.org/dataframe.html) to improve speed