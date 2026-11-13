# -*- coding: utf-8 -*-
"""
13MA 閉環系統 —— 把老哥說的「閉環」做成程式（第 2～4 層）

  第 1 層 交易流程閉環：進場 → 持有 → 出場 → 回到空手（ma13_system.py 已有）
  第 2 層 時間閉環：夜盤 → 開盤跳空 → 第一根 5 分 K → 收盤 → 夜盤…，每天記錄、檢查有沒有漏掉的時段
  第 3 層 週期閉環（共振）：日線定大方向、30 分 K 定波段、5 分 K 抓進出點
  第 4 層 驗證閉環（從尾到頭）：每筆交易記下是哪條規則進場、哪條規則出場，再依規則統計

第 3 層的規則（狀態表）：
  進場訊號：5 分 K「多頭成立／空頭成立」（站上／跌破 13MA＋KD 過 50＋MACD 柱過 0，三者一起）
  依當時日線、30 分 K 的方向決定單子的性質：
    日線、30 分、5 分同向        → 波段單（可留倉；30 分 K 反向穿越 13MA 或日線翻轉才出）
    日線同向、30 分未同步         → 當沖單
    30 分同向、逆日線             → 當沖單
    日線、30 分都反向（只有 5 分）→ 當沖單（可用 --no-counter 不做）
  當沖單：5 分 K 反向穿越 13MA 出場，最晚收盤前最後一根 K 棒平倉，不留倉
  所有單子：獲利從最高回吐超過一半就砍；可加固定停損

  只用「已經收盤」的大週期 K 棒判斷方向，不偷看未來：
    日線方向用前一個交易日收盤的狀態；30 分 K 用最近一根已收完的狀態。

用法：
  python ma13_closed_loop.py                         # 加權指數（Yahoo：日線 1 年＋5 分 K 近 60 天）
  python ma13_closed_loop.py --symbol 2330.TW --long-only
  python ma13_closed_loop.py --csv TXF_1m.csv        # 自備 1 分或 5 分 K（可含夜盤，用來記錄夜盤）

  夜盤資料：Yahoo 沒有台指期夜盤。可用 sinopac_data.fetch_minutes(api, "TXFR1", 起, 迄,
  day_session_only=False) 抓含夜盤的 1 分 K，存成 CSV 後用 --csv 讀入。

輸出（存在「交易記錄」資料夾，Excel 可直接開）：
  每日時間記錄_<代號>.csv   第 2 層：每天的夜盤、跳空、第一根 5 分 K、收盤、三週期狀態、當日交易、缺漏檢查
  交易明細_<代號>.csv       第 4 層：每筆交易的進出場規則、三週期狀態、最大有利／不利、損益
  規則統計_<代號>.csv       第 4 層：依「單子類型／進場規則／出場規則」統計勝率與損益

注意：教學用的規則化程式，以 5 分 K 收盤價成交；除 --cost 外未計手續費、稅、滑價，不構成投資建議。
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ma13_system import HALF_RETRACE, OPEN_ENERGY_MULT, add_indicators, resample_ohlc

DAY_SESSION = (dt.time(8, 45), dt.time(13, 45))   # 期貨日盤；股票、指數 9:00–13:30 也落在其中
DIR_TXT = {1: "多", -1: "空", 0: "—"}
OHLC = ["Open", "High", "Low", "Close"]


# ───────────────────────── 資料 ─────────────────────────
def fetch_yahoo(symbol: str, interval: str, period: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("Asia/Taipei").tz_localize(None)
    return df[OHLC].dropna()


def bar_minutes(df: pd.DataFrame) -> int:
    d = df.index.to_series().diff().dropna()
    d = d[d <= pd.Timedelta(minutes=60)]
    return int(d.median().total_seconds() // 60) if len(d) else 5


def split_sessions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """分成日盤與夜盤（夜盤：日盤以外的時段，例如台指期 15:00～隔天 05:00）"""
    t = df.index.time
    day = (t >= DAY_SESSION[0]) & (t < DAY_SESSION[1])
    return df[day], df[~day]


def daily_from_intraday(day: pd.DataFrame) -> pd.DataFrame:
    d = day.groupby(day.index.date).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
    d.index = pd.to_datetime(d.index)
    return d


def load(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """回傳（日盤 5 分 K、夜盤 K、日 K、資料說明）"""
    if args.csv:
        raw = pd.read_csv(args.csv, parse_dates=[0], index_col=0)[OHLC].dropna().sort_index()
        day, night = split_sessions(raw)
        mins = bar_minutes(day)
        d5 = day if mins == 5 else resample_ohlc(day, 5)
        daily = daily_from_intraday(day)
        src = f"{args.csv}（{mins} 分 K，日盤 {len(day)} 根、夜盤 {len(night)} 根）"
        if len(daily) < 40:
            print(f"⚠ 只有 {len(daily)} 天資料，日線 13MA／MACD 要約 40 天才穩定，前面幾天的日線方向會是「—」。")
    else:
        d5 = fetch_yahoo(args.symbol, "5m", "60d")
        daily = fetch_yahoo(args.symbol, "1d", "1y")
        night = pd.DataFrame(columns=OHLC)
        src = f"{args.symbol}（Yahoo：5 分 K 近 60 天、日 K 1 年；無夜盤）"
    if d5.empty or daily.empty:
        sys.exit("抓不到資料，請確認代號或 CSV 檔。")
    return d5, night, daily, src


# ───────────────────────── 商品類型、跳動單位、當沖濾網 ─────────────────────────
KIND_TXT = {"stock": "股票", "etf": "ETF", "index": "指數", "futures": "期貨"}


def instrument_kind(code: str) -> str:
    """由代號猜商品類型：index（^TWII、TSE001）、futures（TXFR1、MXFR1）、etf（00 開頭）、stock"""
    c = code.strip().upper()
    if c.startswith("^") or (c[:3] in ("TSE", "OTC") and c[3:].isdigit()):
        return "index"
    base = c.split(".")[0].split(" ")[0]
    if base[:4].isdigit():                      # 2330、0050、00631L
        return "etf" if base.startswith("00") else "stock"
    return "futures"


def tick_size(price: float, kind: str) -> float:
    """台灣最小跳動單位（指數、台指期以 1 點計）"""
    if kind in ("index", "futures"):
        return 1.0
    if kind == "etf":
        return 0.01 if price < 50 else 0.05
    for limit, tick in ((10, 0.01), (50, 0.05), (100, 0.1), (500, 0.5), (1000, 1.0)):
        if price < limit:
            return tick
    return 5.0


def bar_ticks(d5: pd.DataFrame, tick: float, recent_days: int = 20) -> tuple[float, float]:
    """最近 recent_days 天 5 分K 的平均振幅，以及等於幾個跳動單位"""
    days = sorted(set(d5.index.date))[-recent_days:]
    recent = d5[pd.Index(d5.index.date).isin(days)]
    rng = float((recent["High"] - recent["Low"]).mean())
    return rng, rng / tick if tick else np.inf


def daytrade_check(d5: pd.DataFrame, kind: str, min_ticks: float = 3.0,
                   tick: float | None = None) -> dict:
    """5 分K 平均振幅不到 min_ticks 檔 → 波動太小，13MA 會被一檔來回洗，不做當沖單"""
    tick = tick or tick_size(float(d5["Close"].iloc[-1]), kind)
    rng, n = bar_ticks(d5, tick)
    return {"tick": tick, "avg_range": rng, "ticks": n, "allow_daytrade": n >= min_ticks,
            "min_ticks": min_ticks}


# ───────────────────────── 第 3 層：三週期方向 ─────────────────────────
def direction(df: pd.DataFrame) -> np.ndarray:
    """13MA 之上＝多(1)，之下＝空(-1)，還算不出來＝0"""
    d = np.sign(df["Close"] - df["MA13"]).fillna(0).astype(int).to_numpy()
    return d


def align_closed(base_end: pd.Series, higher: pd.DataFrame, end_offset: pd.Timedelta,
                 values: dict[str, np.ndarray]) -> pd.DataFrame:
    """把大週期「已收盤」的狀態對到每根 5 分 K 的收盤時間"""
    hi = pd.DataFrame({"end": higher.index + end_offset, **values}).sort_values("end")
    m = pd.merge_asof(base_end.to_frame("end"), hi, on="end", direction="backward")
    return m.drop(columns="end")


def build_frame(d5: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """5 分 K 加上指標，並附上同一時間點的 30 分 K、日線方向"""
    f5 = add_indicators(d5)
    f30 = add_indicators(resample_ohlc(d5, 30))
    fd = add_indicators(daily)

    end5 = pd.Series(f5.index + pd.Timedelta(minutes=5))
    a30 = align_closed(end5, f30, pd.Timedelta(minutes=30), {"三十分": direction(f30)})
    # 日 K 的索引是當天 00:00；要到「隔天」才算收盤完成 → 位移一天，盤中只會看到前一天的日線
    ad = align_closed(end5, fd, pd.Timedelta(days=1), {"日線": direction(fd)})

    f5["五分"] = direction(f5)
    f5["三十分"] = a30["三十分"].fillna(0).astype(int).to_numpy()
    f5["日線"] = ad["日線"].fillna(0).astype(int).to_numpy()
    dates = pd.Series(f5.index.date, index=f5.index)
    f5["當日最後一根"] = dates.ne(dates.shift(-1)).to_numpy()
    f5["當日第一根"] = dates.ne(dates.shift(1)).to_numpy()
    return f5, f30, fd


def classify(side: int, d: int, m30: int) -> tuple[str, str]:
    """依狀態表決定單子類型與進場規則"""
    if d == side and m30 == side:
        return "波段", "三週期共振"
    if d == side:
        return "當沖", "順日線・30分未同步"
    if m30 == side:
        return "當沖", "順30分・逆日線"
    return "當沖", "只有5分（逆日線與30分）"


# ───────────────────────── 第 4 層：交易記錄 ─────────────────────────
@dataclass
class Trade:
    編號: int
    類型: str
    方向: str
    進場規則: str
    進場時間: pd.Timestamp
    進場價: float
    進場時日線: str
    進場時30分: str
    進場時5分K: float
    進場時13MA: float
    出場時間: pd.Timestamp | None = None
    出場價: float = np.nan
    出場規則: str = ""
    持有K棒: int = 0
    最大有利: float = 0.0
    最大不利: float = 0.0
    成本: float = 0.0

    @property
    def sign(self) -> int:
        return 1 if self.方向 == "多" else -1

    def row(self) -> dict:
        pnl = (self.出場價 - self.進場價) * self.sign - self.成本
        return {
            "編號": self.編號, "日期": self.進場時間.date(), "類型": self.類型, "方向": self.方向,
            "進場規則": self.進場規則, "進場時間": self.進場時間, "進場價": round(self.進場價, 2),
            "進場時日線": self.進場時日線, "進場時30分": self.進場時30分,
            "進場時13MA": round(self.進場時13MA, 2),
            "出場時間": self.出場時間, "出場價": round(self.出場價, 2), "出場規則": self.出場規則,
            "持有K棒": self.持有K棒, "留倉": self.出場時間.date() != self.進場時間.date(),
            "最大有利": round(self.最大有利, 2), "最大不利": round(-self.最大不利, 2),
            "損益點": round(pnl, 2), "損益%": round(pnl / self.進場價 * 100, 3),
            "賺賠": "賺" if pnl > 0 else "賠",
        }


def run(f5: pd.DataFrame, long_only: bool = False, counter: bool = True, stop: float | None = None,
        half_min_pct: float = 0.1, cost: float = 0.0, half_min_pts: float | None = None,
        allow_daytrade: bool = True, half_floor: float = 0.0) -> pd.DataFrame:
    """
    half_min_pts 有給（點數）就用它當「回吐一半」的啟動門檻，否則用進場價的 half_min_pct %。
    allow_daytrade=False：波動太小的商品不做當沖單，只做三週期共振的波段單。
    half_floor：「回吐一半」門檻的下限（例如 3 檔），避免賺一檔、回一檔就被洗出場。
    """
    trades: list[Trade] = []
    pos: Trade | None = None
    best = np.nan
    cols = {c: f5[c].to_numpy() for c in
            ["Open", "High", "Low", "Close", "MA13", "五分", "三十分", "日線",
             "多頭成立", "空頭成立", "當日最後一根"]}
    idx = f5.index

    for i, ts in enumerate(idx):
        price, ma = cols["Close"][i], cols["MA13"][i]
        if np.isnan(ma):
            continue
        d, m30, last = cols["日線"][i], cols["三十分"][i], cols["當日最後一根"][i]

        # ── 持有中：檢查出場（每一種情況都有下一步）
        if pos is not None:
            s = pos.sign
            pos.持有K棒 += 1
            fav = (cols["High"][i] - pos.進場價) if s == 1 else (pos.進場價 - cols["Low"][i])
            adv = (pos.進場價 - cols["Low"][i]) if s == 1 else (cols["High"][i] - pos.進場價)
            pos.最大有利, pos.最大不利 = max(pos.最大有利, fav), max(pos.最大不利, adv)
            best = max(best, price) if s == 1 else min(best, price)
            gain, max_gain = (price - pos.進場價) * s, (best - pos.進場價) * s

            reason = ""
            if pos.類型 == "波段":
                if m30 == -s:
                    reason = f"30分{'跌破' if s == 1 else '站上'}13MA"
                elif d == -s:
                    reason = "日線翻轉"
            elif (price - ma) * s < 0:
                reason = f"5分{'跌破' if s == 1 else '站上'}13MA"
            thr = max(half_min_pts if half_min_pts is not None else pos.進場價 * half_min_pct / 100,
                      half_floor)
            if not reason and max_gain > thr and gain < max_gain * HALF_RETRACE:
                reason = "回吐超過一半"
            if not reason and stop is not None and gain <= -stop:
                reason = f"停損{stop:g}點"
            if not reason and pos.類型 == "當沖" and last:
                reason = "當沖收盤平倉"
            if reason:
                pos.出場時間, pos.出場價, pos.出場規則 = ts, price, reason
                trades.append(pos)
                pos = None

        # ── 空手：檢查進場（當天最後一根不進場，避免當沖單無法平倉）
        if pos is None and not last:
            side = 1 if cols["多頭成立"][i] else (-1 if cols["空頭成立"][i] else 0)
            if side == -1 and long_only:
                side = 0
            if side:
                typ, rule = classify(side, d, m30)
                if (rule.startswith("只有5分") and not counter) or (typ == "當沖" and not allow_daytrade):
                    continue
                pos = Trade(len(trades) + 1, typ, DIR_TXT[side], rule, ts, price,
                            DIR_TXT[d], DIR_TXT[m30], price, ma, 成本=cost)
                best = price

    if pos is not None:
        pos.出場時間, pos.出場價, pos.出場規則 = idx[-1], cols["Close"][-1], "期末未平倉（以最後價計）"
        trades.append(pos)
    return pd.DataFrame([t.row() for t in trades])


def rule_stats(tr: pd.DataFrame) -> pd.DataFrame:
    """從尾到頭：哪一條規則賺、哪一條規則賠"""
    if tr.empty:
        return pd.DataFrame()
    out = []
    for key in (["類型"], ["類型", "進場規則"], ["出場規則"], ["方向"]):
        g = tr.groupby(key)
        s = pd.DataFrame({
            "筆數": g.size(),
            "勝率%": g["損益點"].apply(lambda x: (x > 0).mean() * 100).round(1),
            "總損益點": g["損益點"].sum().round(1),
            "平均損益點": g["損益點"].mean().round(1),
            "平均最大有利": g["最大有利"].mean().round(1),
            "平均最大不利": g["最大不利"].mean().round(1),
        }).reset_index()
        s.insert(0, "分組", "／".join(key))
        s.insert(1, "項目", s[key].astype(str).agg("／".join, axis=1))
        out.append(s.drop(columns=key))
    return pd.concat(out, ignore_index=True)


# ───────────────────────── 第 2 層：每日時間記錄 ─────────────────────────
def daily_journal(f5: pd.DataFrame, night: pd.DataFrame, fd: pd.DataFrame, f30: pd.DataFrame,
                  trades: pd.DataFrame, expect_night: bool = True) -> pd.DataFrame:
    """expect_night=False（股票、ETF、指數本來就沒有夜盤）時，不把「沒有夜盤」算成缺漏"""
    days = list(f5.groupby(f5.index.date))
    if not days:
        return pd.DataFrame()
    first_times = pd.Series([g.index[0].time() for _, g in days])
    last_times = pd.Series([g.index[-1].time() for _, g in days])
    normal_open, normal_last = first_times.mode()[0], last_times.mode()[0]
    has_night = len(night) > 0
    d_dir = pd.Series(direction(fd), index=fd.index.date)
    m30_last = pd.Series(direction(f30), index=f30.index).groupby(f30.index.date).last()

    rows = []
    for j, (day, g) in enumerate(days):
        first = g.iloc[0]
        missing = []
        if g.index[0].time() != normal_open:
            missing.append(f"開盤資料缺（第一根 {g.index[0]:%H:%M}）")
        if g.index[-1].time() != normal_last:
            missing.append(f"收盤資料缺（最後一根 {g.index[-1]:%H:%M}）")
        row = {"日期": day, "星期": "一二三四五六日"[day.weekday()]}

        prev_close = np.nan
        if j > 0:
            prev_day, prev = days[j - 1]
            prev_close = prev["Close"].iloc[-1]
            row["昨收"] = prev_close
            # 夜盤：昨天日盤收盤後到今天日盤開盤前
            if has_night:
                lo = pd.Timestamp.combine(prev_day, DAY_SESSION[1])
                hi = pd.Timestamp.combine(day, DAY_SESSION[0])
                n = night[(night.index >= lo) & (night.index < hi)]
                if len(n):
                    row.update({"夜盤高": n["High"].max(), "夜盤低": n["Low"].min(), "夜盤收": n["Close"].iloc[-1],
                                "夜盤漲跌": n["Close"].iloc[-1] - prev_close,
                                "夜盤收→開盤": first["Open"] - n["Close"].iloc[-1]})
                elif expect_night:
                    missing.append("夜盤缺")
            elif expect_night:
                missing.append("無夜盤資料")
        else:
            missing.append("第一天：無昨收可比")

        gap = first["Open"] - prev_close
        rng5 = first["High"] - first["Low"]
        day_rng = g["High"].max() - g["Low"].min()
        row.update({
            "開盤": first["Open"], "跳空點": gap, "跳空%": gap / prev_close * 100,
            "跳空方向": "—" if np.isnan(gap) else ("上" if gap > 0 else ("下" if gap < 0 else "平")),
            "缺口大小": "—" if np.isnan(gap) else ("大（看力量・出貨）" if abs(gap) > rng5 else "小（看方向・進貨）"),
            "第一根5分K振幅": rng5, f"推估幅度(×{OPEN_ENERGY_MULT})": rng5 * OPEN_ENERGY_MULT,
            "實際幅度": day_rng, "推估是否涵蓋": day_rng <= rng5 * OPEN_ENERGY_MULT,
        })
        if not np.isnan(gap) and gap != 0:
            half = prev_close + gap / 2
            reach = g["Low"].min() if gap > 0 else g["High"].max()
            filled_half = reach <= half if gap > 0 else reach >= half
            filled_full = reach <= prev_close if gap > 0 else reach >= prev_close
            row["缺口一半價"] = half
            row["缺口回補"] = "全補" if filled_full else ("補一半" if filled_half else "未補")
        row.update({
            "最高": g["High"].max(), "最低": g["Low"].min(), "收盤": g["Close"].iloc[-1],
            "收盤時日線": DIR_TXT.get(int(d_dir.get(day, 0)), "—"),
            "收盤時30分": DIR_TXT.get(int(m30_last.get(day, 0)), "—"),
            "收盤時5分": DIR_TXT[int(g["五分"].iloc[-1])],
        })
        if not trades.empty:
            t = trades[trades["日期"] == day]
            row["當日交易筆數"] = len(t)
            row["當日損益點"] = round(t["損益點"].sum(), 1)
        row["時間銜接檢查"] = "完整" if not missing else "；".join(missing)
        rows.append(row)
    out = pd.DataFrame(rows)
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(2)
    return out


# ───────────────────────── 主程式 ─────────────────────────
def summary(tr: pd.DataFrame) -> str:
    if tr.empty:
        return "沒有交易訊號。"
    p = tr["損益點"]
    w, l = p[p > 0], p[p <= 0]
    pf = w.sum() / -l.sum() if l.sum() < 0 else np.inf
    eq = p.cumsum()
    return (f"交易 {len(p)} 筆｜勝率 {len(w) / len(p):.1%}｜總損益 {p.sum():+,.1f} 點｜獲利因子 {pf:.2f}\n"
            f"平均賺 {w.mean() if len(w) else 0:+,.1f}｜平均賠 {l.mean() if len(l) else 0:+,.1f}"
            f"｜最大單筆虧損 {p.min():+,.1f}｜最大回撤 {(eq - eq.cummax()).min():+,.1f}")


def now_state(f5: pd.DataFrame) -> str:
    r = f5.iloc[-1]
    d, m30, m5 = int(r["日線"]), int(r["三十分"]), int(r["五分"])
    line = (f"最新 5 分 K {f5.index[-1]}  收 {r['Close']:,.2f}  5分13MA {r['MA13']:,.2f}\n"
            f"  日線（前一日收盤）{DIR_TXT[d]}｜30 分 K {DIR_TXT[m30]}｜5 分 K {DIR_TXT[m5]}")
    if d == m30 == m5 != 0:
        line += f"\n  → 三週期共振（{DIR_TXT[d]}）：可抱波段單"
    elif d == m30 != 0:
        line += f"\n  → 日線、30 分同為{DIR_TXT[d]}，5 分 K 逆向：波段單減碼或觀望；逆向只能做當沖"
    else:
        line += "\n  → 週期不一致：只做當沖，收盤前一定平倉"
    return line


def safe_name(s: str) -> str:
    return re.sub(r"[^\w\-]+", "_", s).strip("_") or "data"


def main():
    p = argparse.ArgumentParser(description="13MA 閉環系統：三週期共振＋每日時間記錄＋交易記錄")
    p.add_argument("--symbol", default="^TWII", help="Yahoo 代號，如 ^TWII、2330.TW")
    p.add_argument("--csv", help="自備 1 分或 5 分 K（欄位 Datetime,Open,High,Low,Close；可含夜盤）")
    p.add_argument("--long-only", action="store_true", help="只做多（個股）")
    p.add_argument("--no-counter", action="store_true", help="日線與 30 分都反向時不做（不逆勢當沖）")
    p.add_argument("--stop", type=float, help="固定停損點數")
    p.add_argument("--half-min-pct", type=float, default=0.1,
                   help="獲利超過進場價的幾 %%，才啟用「回吐一半就砍」（預設 0.1%%，避免小波動就出場）")
    p.add_argument("--cost", type=float, default=0.0, help="每筆來回成本（點）")
    p.add_argument("--kind", choices=["auto", *KIND_TXT], default="auto",
                   help="商品類型（決定跳動單位、要不要檢查夜盤）；auto 依代號判斷，CSV 有夜盤就當期貨")
    p.add_argument("--tick", type=float, help="最小跳動單位（不填就依台灣規則自動算）")
    p.add_argument("--min-ticks", type=float, default=3.0,
                   help="5 分K 平均振幅至少幾檔才做當沖單（預設 3；設 0 關閉濾網）")
    p.add_argument("--out", default="交易記錄", help="輸出資料夾")
    p.add_argument("--show", type=int, default=10, help="畫面顯示最近幾筆交易／幾天記錄")
    args = p.parse_args()

    d5, night, daily, src = load(args)
    f5, f30, fd = build_frame(d5, daily)
    print(f"══ 13MA 閉環系統 ══ {src}\n")

    if args.kind != "auto":
        kind = args.kind
    elif args.csv:
        kind = "futures" if len(night) else "index"
    else:
        kind = instrument_kind(args.symbol)
    chk = daytrade_check(d5, kind, args.min_ticks, args.tick)

    print("【目前三週期狀態】")
    print(now_state(f5))
    print(f"\n【商品】{KIND_TXT[kind]}｜跳動單位 {chk['tick']:g}｜最近 20 天 5 分K 平均振幅 "
          f"{chk['avg_range']:,.2f}（約 {chk['ticks']:.1f} 檔）")
    if chk["allow_daytrade"]:
        print(f"  → 振幅 ≥ {args.min_ticks:g} 檔：波段單與當沖單都做")
    else:
        print(f"  → 振幅不到 {args.min_ticks:g} 檔：5 分K 的 13MA 會被一兩檔來回洗，這檔不適合 5 分K 當沖，只做波段單")

    trades = run(f5, long_only=args.long_only, counter=not args.no_counter, stop=args.stop,
                 half_min_pct=args.half_min_pct, cost=args.cost, allow_daytrade=chk["allow_daytrade"],
                 half_floor=chk["tick"] * args.min_ticks)
    stats = rule_stats(trades)
    journal = daily_journal(f5, night, fd, f30, trades, expect_night=kind == "futures")

    os.makedirs(args.out, exist_ok=True)
    tag = safe_name(os.path.splitext(os.path.basename(args.csv))[0] if args.csv else args.symbol)
    paths = {
        "每日時間記錄": os.path.join(args.out, f"每日時間記錄_{tag}.csv"),
        "交易明細": os.path.join(args.out, f"交易明細_{tag}.csv"),
        "規則統計": os.path.join(args.out, f"規則統計_{tag}.csv"),
    }
    journal.to_csv(paths["每日時間記錄"], index=False, encoding="utf-8-sig")
    trades.to_csv(paths["交易明細"], index=False, encoding="utf-8-sig")
    stats.to_csv(paths["規則統計"], index=False, encoding="utf-8-sig")

    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.unicode.east_asian_width", True):
        print(f"\n【第 2 層・每日時間記錄】（最近 {args.show} 天）")
        cols = ["日期", "昨收", "開盤", "跳空點", "缺口大小", "第一根5分K振幅", "實際幅度",
                "缺口回補", "收盤", "收盤時日線", "收盤時30分", "當日損益點", "時間銜接檢查"]
        print(journal[[c for c in cols if c in journal]].tail(args.show).to_string(index=False))
        broken = journal[journal["時間銜接檢查"] != "完整"]
        print(f"  時間銜接：{len(journal) - len(broken)}/{len(journal)} 天完整"
              + (f"；常見缺漏：{broken['時間銜接檢查'].value_counts().index[0]}" if len(broken) else ""))

        print("\n【第 3 層・三週期共振回測】（5 分 K 收盤價成交）")
        print(summary(trades))
        if not trades.empty:
            cols = ["編號", "類型", "方向", "進場規則", "進場時間", "進場價", "出場時間", "出場價",
                    "出場規則", "損益點"]
            print(trades[cols].tail(args.show).to_string(index=False))

        print("\n【第 4 層・規則統計（從尾到頭）】")
        if not stats.empty:
            print(stats.drop(columns=["平均最大有利", "平均最大不利"]).to_string(index=False))

    print("\n已存檔：")
    for k, v in paths.items():
        print(f"  {k}：{v}")


if __name__ == "__main__":
    main()
