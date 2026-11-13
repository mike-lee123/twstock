# -*- coding: utf-8 -*-
"""
13MA 的魔力 —— 把錄音中老哥的操作規則整理成程式

規則來源（老哥錄音談話整理）：
  1. 只信開盤與收盤：開盤跳空＝非線性能量；能量＝缺口＋第一根 5 分 K 大小
     第一根 5 分 K 振幅 × 6 ≈ 當天可能的波動幅度
  2. 13MA 是多空開關：站上 13MA 且 KD>50、MACD 過 0 → 做多；跌破 13MA → 做空
  3. K 線三根法則：一根過前兩根高 → 開始動；一根破前兩根低 → 開始跌
  4. 回檔超過一半就砍：獲利從最高回吐超過 50% 出場
  5. 費氏座標：0.382 / 0.5 / 0.618 回檔，費氏數列關卡（80→130→210…）
  6. 飆股：KD>80 且低點不破 → 目標價 ×3（極端條件，僅作標記）
  7. 缺口要量化記錄：跳多少點、補了一半還是全補

用法：
  python ma13_system.py                              # 台股加權指數 30 分 K（近 60 天）
  python ma13_system.py --symbol 2330.TW --interval 1d --period 2y --long-only
  python ma13_system.py --interval 5m --period 30d   # 5 分 K
  python ma13_system.py --mode mtf --cost 3          # 30 分K定方向＋5 分K進場，每筆扣 3 點
  python ma13_system.py --csv mydata.csv --interval 5m  # 自備資料：欄位 Datetime,Open,High,Low,Close
                                                     # （CSV 為 5 分 K 時才會產生開盤劇情）

注意：這是教學用的規則化程式，未計手續費、滑價與稅，不構成投資建議。
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MA_LEN = 13
OPEN_ENERGY_MULT = 6          # 第一根 5 分 K × 6 ≈ 當日幅度
HALF_RETRACE = 0.5            # 回吐一半就砍
FIB_RATIOS = (0.382, 0.5, 0.618)


# ───────────────────────── 指標 ─────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA13"] = df["Close"].rolling(MA_LEN).mean()

    # 台灣常用 KD(9,3,3)
    low9 = df["Low"].rolling(9).min()
    high9 = df["High"].rolling(9).max()
    rsv = ((df["Close"] - low9) / (high9 - low9).replace(0, np.nan) * 100).fillna(50)
    k = np.full(len(df), 50.0)
    d = np.full(len(df), 50.0)
    for i in range(1, len(df)):
        k[i] = k[i - 1] * 2 / 3 + rsv.iloc[i] / 3
        d[i] = d[i - 1] * 2 / 3 + k[i] / 3
    df["K"], df["D"] = k, d

    # MACD(12,26,9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["MACD"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["OSC"] = df["DIF"] - df["MACD"]

    # K 線三根法則
    prev2_high = df["High"].shift(1).rolling(2).max()
    prev2_low = df["Low"].shift(1).rolling(2).min()
    df["過兩根高"] = df["Close"] > prev2_high
    df["破兩根低"] = df["Close"] < prev2_low

    # 13MA 多空狀態：三個條件一起過，不是單一條件
    bull = (df["Close"] > df["MA13"]) & (df["K"] > 50) & (df["OSC"] > 0)
    bear = (df["Close"] < df["MA13"]) & (df["K"] < 50) & (df["OSC"] < 0)
    df["多頭成立"] = bull & ~bull.shift(1, fill_value=False)
    df["空頭成立"] = bear & ~bear.shift(1, fill_value=False)
    df["跌破13MA"] = (df["Close"] < df["MA13"]) & (df["Close"].shift(1) >= df["MA13"].shift(1))
    df["站上13MA"] = (df["Close"] > df["MA13"]) & (df["Close"].shift(1) <= df["MA13"].shift(1))

    # 飆股標記：KD>80 且低點沒破前低
    df["飆股標記"] = (df["K"] > 80) & (df["Low"] >= df["Low"].shift(1))
    return df


# ───────────────────────── 費氏座標 ─────────────────────────
def fib_retracements(low: float, high: float) -> dict[str, float]:
    """回檔關卡：從高點回 0.382 / 0.5 / 0.618；延伸：一倍、1.618 倍"""
    rng = high - low
    out = {f"回檔 {r}": high - rng * r for r in FIB_RATIOS}
    out["延伸 1.0（再複製一倍）"] = high + rng
    out["延伸 1.618"] = high + rng * 0.618 + rng
    return out


def fib_sequence_levels(price: float, count: int = 4) -> list[float]:
    """費氏數列座標：例 80 → 130 → 210 → 340（數列 ×10 的冪次）"""
    scale = 10 ** int(np.floor(np.log10(price)))      # 80 → 10；43,760 → 10,000
    seq = [1, 2]
    while len(seq) < 30:
        seq.append(seq[-1] + seq[-2])
    return [f * scale for f in seq if f * scale > price][:count]


# ───────────────────────── 開盤劇情（日內資料） ─────────────────────────
@dataclass
class OpeningReport:
    date: object
    prev_close: float
    open: float
    gap: float
    gap_pct: float
    first_bar_range: float
    projected_range: float
    half_fill_level: float
    half_filled: bool
    full_filled: bool
    day_high: float
    day_low: float
    close: float

    def text(self) -> str:
        direction = "往上跳空" if self.gap > 0 else ("往下跳空" if self.gap < 0 else "平盤開出")
        size = "大缺口（看力量→出貨）" if abs(self.gap) > self.first_bar_range else "小缺口（看方向→進貨）"
        fill = "已全部回補" if self.full_filled else ("補了一半" if self.half_filled else "未回補")
        return (
            f"日期 {self.date}｜昨收 {self.prev_close:,.0f} → 開盤 {self.open:,.0f}\n"
            f"  {direction} {self.gap:+,.0f} 點（{self.gap_pct:+.2f}%），{size}\n"
            f"  第一根 5 分 K 振幅 {self.first_bar_range:,.0f} 點 × {OPEN_ENERGY_MULT} "
            f"≈ 今日可能幅度 {self.projected_range:,.0f} 點\n"
            f"  缺口一半價位 {self.half_fill_level:,.0f}，回補狀態：{fill}\n"
            f"  當日 高 {self.day_high:,.0f} / 低 {self.day_low:,.0f} / 收 {self.close:,.0f}"
            f"（實際幅度 {self.day_high - self.day_low:,.0f} 點）"
        )


def opening_reports(intraday: pd.DataFrame) -> list[OpeningReport]:
    """每天一筆：跳空大小、第一根 K 能量、缺口回補（缺口要量化記錄）"""
    reports = []
    days = list(intraday.groupby(intraday.index.date))
    for (prev_day, prev), (day, cur) in zip(days, days[1:]):
        prev_close = prev["Close"].iloc[-1]
        first = cur.iloc[0]
        gap = first["Open"] - prev_close
        half = prev_close + gap / 2
        if gap > 0:
            half_filled = cur["Low"].min() <= half
            full_filled = cur["Low"].min() <= prev_close
        elif gap < 0:
            half_filled = cur["High"].max() >= half
            full_filled = cur["High"].max() >= prev_close
        else:
            half_filled = full_filled = True
        rng = first["High"] - first["Low"]
        reports.append(OpeningReport(
            date=day, prev_close=prev_close, open=first["Open"], gap=gap,
            gap_pct=gap / prev_close * 100, first_bar_range=rng,
            projected_range=rng * OPEN_ENERGY_MULT, half_fill_level=half,
            half_filled=half_filled, full_filled=full_filled,
            day_high=cur["High"].max(), day_low=cur["Low"].min(), close=cur["Close"].iloc[-1],
        ))
    return reports


# ───────────────────────── 回測 ─────────────────────────
@dataclass
class Trade:
    side: str
    entry_time: object
    entry: float
    exit_time: object = None
    exit: float = np.nan
    reason: str = ""
    cost: float = 0.0             # 每筆來回成本（點）：手續費＋稅＋滑價

    @property
    def pnl(self) -> float:
        return (self.exit - self.entry) * (1 if self.side == "多" else -1) - self.cost


@dataclass
class Backtest:
    trades: list[Trade] = field(default_factory=list)

    def summary(self) -> str:
        if not self.trades:
            return "沒有交易訊號。"
        pnl = np.array([t.pnl for t in self.trades])
        wins = pnl[pnl > 0]
        losses = pnl[pnl <= 0]
        pf = wins.sum() / -losses.sum() if losses.sum() < 0 else np.inf
        return (
            f"交易次數 {len(pnl)}｜勝率 {len(wins) / len(pnl):.1%}｜總損益 {pnl.sum():+,.1f} 點\n"
            f"平均賺 {wins.mean() if len(wins) else 0:+,.1f}｜平均賠 {losses.mean() if len(losses) else 0:+,.1f}"
            f"｜獲利因子 {pf:.2f}｜最大單筆虧損 {pnl.min():+,.1f}"
        )

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "方向": t.side, "進場時間": t.entry_time, "進場": round(t.entry, 2),
            "出場時間": t.exit_time, "出場": round(t.exit, 2),
            "損益": round(t.pnl, 2), "出場原因": t.reason,
        } for t in self.trades])


def backtest(df: pd.DataFrame, long_only: bool = False, stop: float | None = None,
             min_profit_for_half: float = 0.0, cost: float = 0.0) -> Backtest:
    """
    進場：多頭成立（站上 13MA＋KD>50＋MACD 柱>0）做多；空頭成立做空
    出場：(1) 反向跌破／站上 13MA  (2) 獲利回吐超過一半  (3) 固定停損（可選）
    以收盤價成交；cost 為每筆來回成本（點）。
    """
    bt = Backtest()
    pos: Trade | None = None
    best = np.nan
    for ts, row in df.iterrows():
        if np.isnan(row["MA13"]):
            continue
        price = row["Close"]
        if pos is not None:
            sign = 1 if pos.side == "多" else -1
            best = max(best, price) if sign == 1 else min(best, price)
            max_gain = (best - pos.entry) * sign
            gain = (price - pos.entry) * sign
            reason = ""
            if sign == 1 and price < row["MA13"]:
                reason = "跌破13MA"
            elif sign == -1 and price > row["MA13"]:
                reason = "站上13MA"
            elif max_gain > min_profit_for_half and gain < max_gain * HALF_RETRACE:
                reason = "回吐超過一半"
            elif stop is not None and gain <= -stop:
                reason = f"停損 {stop:g}"
            if reason:
                pos.exit_time, pos.exit, pos.reason = ts, price, reason
                bt.trades.append(pos)
                pos = None
        if pos is None:
            if row["多頭成立"]:
                pos, best = Trade("多", ts, price, cost=cost), price
            elif row["空頭成立"] and not long_only:
                pos, best = Trade("空", ts, price, cost=cost), price
    if pos is not None:
        last = df.iloc[-1]
        pos.exit_time, pos.exit, pos.reason = df.index[-1], last["Close"], "期末平倉"
        bt.trades.append(pos)
    return bt


# ───────────────────────── 多週期：30 分 K 定方向、5 分 K 進場 ─────────────────────────
def resample_ohlc(df: pd.DataFrame, minutes: int = 30) -> pd.DataFrame:
    """把日內 K 棒合成大週期，每天從當天第一根 K 棒開始切（台股 9:00、9:30…）"""
    step = pd.Timedelta(minutes=minutes)
    idx = df.index.to_series()
    day_start = idx.groupby(df.index.date).transform("min")
    bucket = day_start + ((idx - day_start) // step) * step
    out = df.groupby(bucket.values).agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    out.index.name = df.index.name
    return out


def backtest_mtf(d_lo: pd.DataFrame, d_hi: pd.DataFrame, lo_minutes: int = 5, hi_minutes: int = 30,
                 bias_mode: str = "ma", exit_mode: str = "lo", long_only: bool = False,
                 stop: float | None = None, min_profit_for_half: float = 0.0,
                 cost: float = 0.0) -> Backtest:
    """
    大週期定方向、小週期進場（兩者都要先跑過 add_indicators）。

    方向（只用「已收盤」的大週期 K 棒，不偷看未來）：
      bias_mode="ma"     大週期收盤在 13MA 之上 → 只做多；之下 → 只做空
      bias_mode="strict" 大週期三條件齊（13MA＋K 50＋MACD 柱 0）才算方向，否則觀望
    進場：方向一致，且小週期三條件齊——在小週期剛成立時，或大週期剛轉向而小週期已成立時。
    出場：exit_mode="lo" 小週期反向穿越 13MA；"hi" 大週期收盤反向穿越 13MA
          另外：大週期方向翻轉、獲利回吐超過一半、固定停損（可選）
    以小週期收盤價成交。
    """
    lo_end = pd.Series(d_lo.index + pd.Timedelta(minutes=lo_minutes), name="end")
    hi = pd.DataFrame({
        "end": d_hi.index + pd.Timedelta(minutes=hi_minutes),
        "hi_close": d_hi["Close"].to_numpy(), "hi_ma": d_hi["MA13"].to_numpy(),
    })
    bull_hi = (d_hi["Close"] > d_hi["MA13"]).to_numpy()
    bear_hi = (d_hi["Close"] < d_hi["MA13"]).to_numpy()
    if bias_mode == "strict":
        bull_hi &= ((d_hi["K"] > 50) & (d_hi["OSC"] > 0)).to_numpy()
        bear_hi &= ((d_hi["K"] < 50) & (d_hi["OSC"] < 0)).to_numpy()
    hi["bias"] = np.where(bull_hi, 1, np.where(bear_hi, -1, 0))
    hi.loc[hi["hi_ma"].isna(), "bias"] = 0
    m = pd.merge_asof(lo_end.to_frame(), hi.sort_values("end"), on="end", direction="backward")
    bias = m["bias"].fillna(0).astype(int).to_numpy()
    hi_close, hi_ma = m["hi_close"].to_numpy(), m["hi_ma"].to_numpy()

    lo_bull = ((d_lo["Close"] > d_lo["MA13"]) & (d_lo["K"] > 50) & (d_lo["OSC"] > 0)).to_numpy()
    lo_bear = ((d_lo["Close"] < d_lo["MA13"]) & (d_lo["K"] < 50) & (d_lo["OSC"] < 0)).to_numpy()
    close, ma = d_lo["Close"].to_numpy(), d_lo["MA13"].to_numpy()
    on_bull, on_bear = d_lo["多頭成立"].to_numpy(), d_lo["空頭成立"].to_numpy()

    bt = Backtest()
    pos: Trade | None = None
    best = np.nan
    for i, ts in enumerate(d_lo.index):
        if np.isnan(ma[i]):
            continue
        price = close[i]
        if pos is not None:
            sign = 1 if pos.side == "多" else -1
            best = max(best, price) if sign == 1 else min(best, price)
            max_gain = (best - pos.entry) * sign
            gain = (price - pos.entry) * sign
            reason = ""
            if bias[i] == -sign:
                reason = f"{hi_minutes}分K轉向"
            elif exit_mode == "lo" and (price - ma[i]) * sign < 0:
                reason = f"{lo_minutes}分K{'跌破' if sign == 1 else '站上'}13MA"
            elif exit_mode == "hi" and not np.isnan(hi_ma[i]) and (hi_close[i] - hi_ma[i]) * sign < 0:
                reason = f"{hi_minutes}分K{'跌破' if sign == 1 else '站上'}13MA"
            elif max_gain > min_profit_for_half and gain < max_gain * HALF_RETRACE:
                reason = "回吐超過一半"
            elif stop is not None and gain <= -stop:
                reason = f"停損 {stop:g}"
            if reason:
                pos.exit_time, pos.exit, pos.reason = ts, price, reason
                bt.trades.append(pos)
                pos = None
        if pos is None:
            flipped = i > 0 and bias[i] != bias[i - 1]
            if bias[i] == 1 and lo_bull[i] and (on_bull[i] or flipped):
                pos, best = Trade("多", ts, price, cost=cost), price
            elif bias[i] == -1 and lo_bear[i] and (on_bear[i] or flipped) and not long_only:
                pos, best = Trade("空", ts, price, cost=cost), price
    if pos is not None:
        pos.exit_time, pos.exit, pos.reason = d_lo.index[-1], close[-1], "期末平倉"
        bt.trades.append(pos)
    return bt


# ───────────────────────── 零股隔日沖：今天收盤買、明天早上賣 ─────────────────────────
ODD_LOT_EXIT_MIN = 10        # 盤中零股約 9:10 第一次撮合 → 取開盤後 10 分鐘的價格

OVERNIGHT_FILTERS = {
    "none": "不過濾（每天都買）",
    "hi_strict": "盤中K 三條件齊（收盤時）",
    "hi_ma": "盤中K 站上 13MA（收盤時）",
    "daily_strict": "日K 三條件齊",
    "daily_below": "日K 在 13MA 下（跌深反彈）",
}
OVERNIGHT_EXITS = {
    "odd": "隔日零股第一盤（約 9:10）",
    "open": "隔日開盤價",
    "gap": "大跳空開盤賣，否則零股第一盤",
    "close": "隔日收盤",
}


def overnight_table(intraday: pd.DataFrame, bar_minutes: int, hi_minutes: int = 30) -> pd.DataFrame:
    """
    每個交易日一列：收盤買進價、隔天各種出場價的毛報酬，以及收盤時各個濾網是否成立。
    intraday 為日內 K（5 分或 60 分）；盤中 K 濾網用 hi_minutes 合成（若與 bar_minutes 相同就直接用）。
    """
    hi = add_indicators(intraday if hi_minutes == bar_minutes else resample_ohlc(intraday, hi_minutes))
    daily = intraday.groupby(intraday.index.date).agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"})
    daily.index = pd.to_datetime(daily.index)
    daily = add_indicators(daily)
    hi_last = hi.groupby(hi.index.date).tail(1)
    hi_last.index = pd.to_datetime(hi_last.index.date)

    rows = []
    days = list(intraday.groupby(intraday.index.date))
    for j, (day, cur) in enumerate(days):
        d = pd.Timestamp(day)
        buy = cur["Close"].iloc[-1]
        h = hi_last.loc[d] if d in hi_last.index else None
        dl = daily.loc[d]
        row = {
            "日期": d.date(), "出場日": None, "買進價": buy,
            "隔日開": np.nan, "隔日零股第一盤": np.nan, "隔日收": np.nan, "跳空%": np.nan,
            "r_open": np.nan, "r_odd": np.nan, "r_close": np.nan,
            "none": True,
            "hi_strict": bool(h is not None and h["Close"] > h["MA13"] and h["K"] > 50 and h["OSC"] > 0),
            "hi_ma": bool(h is not None and h["Close"] > h["MA13"]),
            "daily_strict": bool(dl["Close"] > dl["MA13"] and dl["K"] > 50 and dl["OSC"] > 0),
            "daily_below": bool(dl["Close"] < dl["MA13"]),
        }
        if j + 1 < len(days):                     # 最後一天沒有「隔日」→ 只留濾網狀態（今晚訊號）
            nday, nxt = days[j + 1]
            n_open = nxt["Open"].iloc[0]
            # 9:10 已收盤的最後一根 K 的收盤價；60 分 K 沒有 9:10，退而用第一根收盤（10:00）
            cutoff = nxt.index[0] + pd.Timedelta(minutes=ODD_LOT_EXIT_MIN - bar_minutes)
            odd = nxt.loc[nxt.index <= cutoff, "Close"]
            odd_px = odd.iloc[-1] if len(odd) else nxt["Close"].iloc[0]
            n_close = nxt["Close"].iloc[-1]
            row.update({
                "出場日": nday, "隔日開": n_open, "隔日零股第一盤": odd_px, "隔日收": n_close,
                "跳空%": (n_open / buy - 1) * 100, "r_open": n_open / buy - 1,
                "r_odd": odd_px / buy - 1, "r_close": n_close / buy - 1,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def overnight_returns(tbl: pd.DataFrame, filter_mode: str, exit_mode: str,
                      gap_pct: float = 1.0, cost_pct: float = 0.4) -> pd.DataFrame:
    """依濾網挑出要買的日子，套用出場方式，回傳含淨報酬（%）的明細"""
    t = tbl[tbl[filter_mode] & tbl["r_open"].notna()].copy()
    if exit_mode == "gap":
        big = t["跳空%"] >= gap_pct
        t["毛報酬%"] = np.where(big, t["r_open"], t["r_odd"]) * 100
        t["出場方式"] = np.where(big, "大跳空・開盤賣", "零股第一盤")
    else:
        t["毛報酬%"] = t[f"r_{exit_mode}"] * 100
        t["出場方式"] = OVERNIGHT_EXITS[exit_mode]
    t["淨報酬%"] = t["毛報酬%"] - cost_pct
    return t


def overnight_stats(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"交易次數": 0}
    r = t["淨報酬%"].to_numpy() / 100
    eq = np.cumprod(1 + r)
    win, loss = r[r > 0], r[r <= 0]
    return {
        "交易次數": len(r), "勝率": (r > 0).mean(), "平均毛報酬%": t["毛報酬%"].mean(),
        "平均淨報酬%": r.mean() * 100, "中位數淨報酬%": np.median(r) * 100,
        "累積報酬%": (eq[-1] - 1) * 100, "最大回撤%": ((eq / np.maximum.accumulate(eq)) - 1).min() * 100,
        "獲利因子": win.sum() / -loss.sum() if loss.sum() < 0 else np.inf,
        "最差一筆%": r.min() * 100,
    }


def odd_lot_cost_pct(amount: float, fee_discount: float = 0.6, min_fee: float = 1.0,
                     tax_rate: float = 0.003) -> float:
    """零股來回成本（%）：手續費 0.1425%×折數（每邊有最低手續費）＋賣出證交稅 0.3%（隔日沖不減半）"""
    fee = max(amount * 0.001425 * fee_discount, min_fee)
    return (2 * fee / amount + tax_rate) * 100


def bt_stats(bt: Backtest) -> dict:
    """回測統計（給比較表用）"""
    if not bt.trades:
        return {"交易次數": 0}
    pnl = np.array([t.pnl for t in bt.trades])
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    eq = np.cumsum(pnl)
    return {
        "交易次數": len(pnl), "勝率": len(wins) / len(pnl), "總損益": pnl.sum(),
        "獲利因子": wins.sum() / -losses.sum() if losses.sum() < 0 else np.inf,
        "平均每筆": pnl.mean(), "最大單筆虧損": pnl.min(),
        "最大回撤": (eq - np.maximum.accumulate(eq)).min(),
    }


# ───────────────────────── 目前狀態 ─────────────────────────
def current_state(df: pd.DataFrame) -> str:
    r = df.iloc[-1]
    lines = [f"最新 K 棒 {df.index[-1]}  收 {r['Close']:,.2f}  13MA {r['MA13']:,.2f}"
             f"  K {r['K']:.1f}  D {r['D']:.1f}  MACD柱 {r['OSC']:+.2f}"]
    if r["Close"] > r["MA13"] and r["K"] > 50 and r["OSC"] > 0:
        lines.append("→ 多頭：站上 13MA、KD>50、MACD 過 0，續抱多單，跌破 13MA 再說")
    elif r["Close"] < r["MA13"]:
        lines.append("→ 在 13MA 之下：偏空，等三個條件一起站回才做多")
    else:
        lines.append("→ 站上 13MA 但 KD／MACD 尚未同步，條件不齊，不追")
    if r["過兩根高"]:
        lines.append("→ 本根過前兩根高：開始動（三根法則）")
    if r["破兩根低"]:
        lines.append("→ 本根破前兩根低：開始往下跑（三根法則）")
    if r["飆股標記"]:
        lines.append(f"→ KD>80 且低點沒破：飆股標記，老哥的目標價 ×3 ≈ {r['Close'] * 3:,.2f}（需天天漲停的極端條件）")

    recent = df.tail(60)
    lo, hi = recent["Low"].min(), recent["High"].max()
    lines.append(f"\n近 60 根區間 低 {lo:,.2f} ~ 高 {hi:,.2f} 的費氏關卡：")
    for name, lv in fib_retracements(lo, hi).items():
        lines.append(f"  {name:<18} {lv:,.2f}")
    lines.append("費氏數列座標（往上的關卡）：" +
                 " → ".join(f"{x:,.0f}" if x >= 1 else f"{x:g}" for x in fib_sequence_levels(r["Close"])))
    return "\n".join(lines)


# ───────────────────────── 資料 ─────────────────────────
def load_data(args) -> pd.DataFrame:
    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=[0], index_col=0)
    else:
        import yfinance as yf
        df = yf.download(args.symbol, period=args.period, interval=args.interval,
                         auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("Asia/Taipei").tz_localize(None)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    if df.empty:
        sys.exit("抓不到資料，請確認代號、週期（5m/30m 最多 60 天）或 CSV 檔。")
    return df


def main():
    p = argparse.ArgumentParser(description="13MA 的魔力：規則化交易程式")
    p.add_argument("--symbol", default="^TWII", help="yfinance 代號，如 ^TWII、2330.TW")
    p.add_argument("--interval", default="30m", help="1m/5m/15m/30m/60m/1d")
    p.add_argument("--period", default="60d", help="資料期間，如 30d、60d、2y")
    p.add_argument("--csv", help="改用自備 CSV")
    p.add_argument("--long-only", action="store_true", help="只做多（個股）")
    p.add_argument("--stop", type=float, help="固定停損點數（如期貨 100 點）")
    p.add_argument("--trades", type=int, default=15, help="顯示最近幾筆交易")
    p.add_argument("--opening-days", type=int, default=5, help="顯示最近幾天的開盤劇情")
    p.add_argument("--cost", type=float, default=0.0, help="每筆來回成本（點）")
    p.add_argument("--mode", choices=["single", "mtf"], default="single",
                   help="single＝單一週期；mtf＝30 分K定方向＋5 分K進場（需 5 分K 資料）")
    p.add_argument("--bias", choices=["strict", "ma"], default="strict",
                   help="mtf 方向判斷：strict 三條件齊；ma 只看 30 分K 在 13MA 上／下")
    p.add_argument("--exit", choices=["lo", "hi"], default="lo",
                   help="mtf 出場：lo 5 分K 穿越 13MA；hi 30 分K 穿越 13MA")
    args = p.parse_args()

    df = add_indicators(load_data(args))
    src = args.csv or f"{args.symbol} {args.interval} / {args.period}"
    print(f"══ 13MA 的魔力 ══ 資料：{src}，共 {len(df)} 根 K 棒\n")

    print("【目前狀態】")
    print(current_state(df))

    # 開盤劇情一定用 5 分 K：第一根 5 分 K × 6 才是當日幅度
    intraday = None
    if args.csv:
        if args.interval == "5m":
            intraday = df
    else:
        import yfinance as yf
        intraday = yf.download(args.symbol, period="60d", interval="5m",
                               auto_adjust=False, progress=False)
        if isinstance(intraday.columns, pd.MultiIndex):
            intraday.columns = intraday.columns.get_level_values(0)
        if intraday.index.tz is not None:
            intraday.index = intraday.index.tz_convert("Asia/Taipei").tz_localize(None)
        intraday = intraday[["Open", "High", "Low", "Close"]].dropna()
    if intraday is not None and len(intraday):
        reps = opening_reports(intraday)
        if reps:
            print(f"\n【開盤劇情：跳空與第一根 5 分 K 能量】（最近 {args.opening_days} 天）")
            for rep in reps[-args.opening_days:]:
                print(rep.text())
            hit = np.mean([(r.day_high - r.day_low) <= r.projected_range for r in reps])
            print(f"  ※ 『第一根 5 分 K ×{OPEN_ENERGY_MULT}』涵蓋實際當日幅度的比例：{hit:.0%}（{len(reps)} 天）")

    cost_txt = f"每筆扣 {args.cost:g} 點" if args.cost else "未計成本"
    if args.mode == "mtf":
        if intraday is None or not len(intraday):
            sys.exit("mtf 模式需要 5 分 K 資料（Yahoo 或 --csv 搭配 --interval 5m）。")
        d5 = intraday if args.csv else add_indicators(intraday)
        d30 = add_indicators(resample_ohlc(intraday[["Open", "High", "Low", "Close"]], 30))
        bt = backtest_mtf(d5, d30, bias_mode=args.bias, exit_mode=args.exit,
                          long_only=args.long_only, stop=args.stop, cost=args.cost)
        print(f"\n【回測：30 分K定方向（{args.bias}）＋5 分K進場，出場 {args.exit}】（以 5 分K 收盤價成交，{cost_txt}）")
    else:
        bt = backtest(df, long_only=args.long_only, stop=args.stop, cost=args.cost)
        print(f"\n【回測：13MA 開關＋回吐一半就砍】（以收盤價成交，{cost_txt}）")
    print(bt.summary())
    if bt.trades:
        with pd.option_context("display.width", 160, "display.max_columns", 10,
                               "display.unicode.east_asian_width", True):
            print(bt.table().tail(args.trades).to_string(index=False))


if __name__ == "__main__":
    main()
