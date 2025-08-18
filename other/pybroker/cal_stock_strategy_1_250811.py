'''
2025-8-11
This script adopt the stock strategy 1 to calculate the stock status. relevant link:
https://chatgpt.com/c/689606b4-8224-8323-b8f0-501de2ee4df4
'''

import numpy as np
import pandas as pd

# =============== 基础工具 ===============
def _rolling_linear_slope(y: pd.Series) -> float:
    """窗口内线性回归斜率（x=0..n-1）"""
    n = len(y)
    if n < 2 or y.isna().any():
        return np.nan
    x = np.arange(n, dtype=float)
    a, _ = np.polyfit(x, y.values, 1)
    return a

def sma(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w).mean()

def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev).abs(),
                    (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume 累积量"""
    dir_ = np.sign(close.diff().fillna(0))  # 收盘涨=+1，跌=-1
    return (volume * dir_).fillna(0).cumsum()

def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
    """Money Flow Index 资金流量指标（0-100）"""
    tp = (high + low + close) / 3.0
    raw_flow = tp * volume
    pos_flow = raw_flow.where(tp > tp.shift(1), 0.0)
    neg_flow = raw_flow.where(tp < tp.shift(1), 0.0)
    pos_sum = pos_flow.rolling(window).sum()
    neg_sum = neg_flow.rolling(window).sum().replace(0, np.nan)
    mr = pos_sum / neg_sum
    return 100 - (100 / (1 + mr))

def realized_vol(ret: pd.Series, window: int = 20) -> pd.Series:
    """近window日年化波动（√252）"""
    return ret.rolling(window).std(ddof=0) * np.sqrt(252)

# =============== 量能优先：特征计算 ===============
def compute_features_volume_first(df: pd.DataFrame) -> pd.DataFrame:
    """
    从成交量出发计算：
      - dollar_volume / avg_dollar_vol_60
      - volume_ma_ratio（5日/20日量均比）
      - volume_roc_20（20日成交量变化率）
      - MFI(14)
      - OBV 及 OBV_slope_20
    然后再接：趋势（日线/周线斜率）、波动（ATR/Close）
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"])
    out = []

    for sym, g in df.groupby("symbol", sort=False):
        g = g.copy()
        o, h, l, c, v = g["open"], g["high"], g["low"], g["close"], g["volume"]

        # ---- 成交量/资金相关 ----
        g["dollar_volume"] = c * v
        g["avg_dollar_vol_60"] = g["dollar_volume"].rolling(60).mean()

        vol_ma_5 = v.rolling(5).mean()
        vol_ma_20 = v.rolling(20).mean()
        g["volume_ma_ratio"] = vol_ma_5 / vol_ma_20  # >1 表示近期放量

        g["volume_roc_20"] = v.pct_change(20)       # 20日量变化率
        g["mfi14"] = mfi(h, l, c, v, 14)            # 资金流量指标
        g["obv"] = obv(c, v)
        g["obv_slope_20"] = g["obv"].rolling(20).apply(_rolling_linear_slope, raw=False)

        # ---- 趋势（先日线，再周线）----
        sma20 = sma(c, 20)
        g["sma20"] = sma20
        g["daily_slope_20"] = sma20.rolling(20).apply(_rolling_linear_slope, raw=False)

        # 周线（以周五为一周），对周收盘做6周SMA再取8周斜率
        gw = g.set_index("date").resample("W-FRI").last()
        if len(gw) >= 14:  # 6周SMA + 8周斜率窗口的最小长度
            sma_w = sma(gw["close"], 6)
            weekly_slope = sma_w.rolling(8).apply(_rolling_linear_slope, raw=False)
            wk = weekly_slope.rename("weekly_slope_6w").to_frame().reset_index()
            g = g.merge(wk, left_on="date", right_on="date", how="left")
            g["weekly_slope_6w"] = g["weekly_slope_6w"].ffill()
        else:
            g["weekly_slope_6w"] = np.nan


        # ---- 波动过滤 ----
        g["atr14"] = atr(h, l, c, 14)
        g["atr_over_close"] = g["atr14"] / c.replace(0, np.nan)

        # ---- 风险/仓位用波动 ----
        g["ret"] = c.pct_change()
        g["ann_vol_20"] = realized_vol(g["ret"], 20)

        g["symbol"] = sym
        out.append(g)

    feat = pd.concat(out, ignore_index=True)
    return feat

# =============== 简单周频回测（权重=逆波） ===============
def weekly_rebalance(feat: pd.DataFrame, screened: pd.DataFrame) -> pd.DataFrame:
    """
    用 screened 的 pass_flag 做每周持仓，权重按 ann_vol_20 的倒数分配。
    收益：下周周末收盘 / 本周周末收盘 - 1。
    """
    if screened.empty:
        raise ValueError("无通过筛选的周截面。")

    # 每周末价格表（对齐收益）
    wk_price = (feat.sort_values(["symbol", "date"])
                    .groupby(["symbol", feat["date"].dt.to_period("W-FRI").dt.to_timestamp("W-FRI")])
                    .tail(1)[["symbol", "date", "close"]]
                    .rename(columns={"date": "week", "close": "close_week"}))

    weeks = sorted(screened["week"].unique())
    results = []

    for i in range(len(weeks) - 1):
        wk = weeks[i]
        nxt = weeks[i+1]

        snap = screened[(screened["week"] == wk) & (screened["pass_flag"])].copy()
        if snap.empty:
            results.append({"week": nxt, "port_ret": 0.0, "n_hold": 0})
            continue

        # 权重：逆波（ann_vol_20 倒数）
        snap = snap.dropna(subset=["ann_vol_20"])
        if snap.empty:
            results.append({"week": nxt, "port_ret": 0.0, "n_hold": 0})
            continue

        inv = 1.0 / snap["ann_vol_20"].replace(0, np.nan)
        inv = inv.replace([np.inf, -np.inf], np.nan).dropna()
        snap = snap.loc[inv.index]
        w = inv / inv.sum()
        snap["weight"] = w.values

        # 本周末 & 下周末价格
        cur = wk_price[wk_price["week"] == wk][["symbol", "close_week"]].rename(columns={"close_week": "p0"})
        nxtp = wk_price[wk_price["week"] == nxt][["symbol", "close_week"]].rename(columns={"close_week": "p1"})
        merged = snap[["symbol", "weight"]].merge(cur, on="symbol").merge(nxtp, on="symbol")
        merged["ret_w"] = merged["p1"] / merged["p0"] - 1.0

        port_ret = float((merged["ret_w"] * merged["weight"]).sum()) if not merged.empty else 0.0
        results.append({"week": nxt, "port_ret": port_ret, "n_hold": int(len(snap))})

    rets = pd.DataFrame(results).sort_values("week")
    rets["equity"] = (1 + rets["port_ret"]).cumprod()
    return rets