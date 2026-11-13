# -*- coding: utf-8 -*-
"""
永豐 Shioaji API：登入、商品代號解析、下載 1 分 K，並合成 5/15/30/60 分、日、週 K。

只讀取行情，不下單；登入不需要憑證（CA）。
代號寫法：
  股票／ETF   2330、0050
  指數        TSE001（加權指數）、OTC101（櫃買指數）
  期貨        TXFR1（台指期近月連續）、MXFR1（小台近月）、TXFR2（次月）
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from ma13_system import resample_ohlc

DAY_SESSION = (dt.time(8, 45), dt.time(13, 45))   # 期貨日盤；股票 9:00–13:30 也落在其中


def login(api_key: str, secret_key: str, simulation: bool = False):
    import shioaji as sj
    api = sj.Shioaji(simulation=simulation)
    accounts = api.login(api_key=api_key.strip(), secret_key=secret_key.strip(),
                         subscribe_trade=False)
    try:
        api.fetch_contracts(contract_download=True)
    except Exception:  # noqa: BLE001 — 有些版本 login 時已自動下載
        pass
    return api, accounts


def account_lines(accounts) -> list[str]:
    """帳號只顯示類別與末 4 碼"""
    out = []
    for a in accounts or []:
        kind = type(a).__name__.replace("Account", "") or "帳戶"
        aid = str(getattr(a, "account_id", ""))
        out.append(f"{kind}　***{aid[-4:]}" if aid else kind)
    return out


def resolve_contract(api, code: str):
    """依代號找合約：指數 TSE/OTC 開頭、期貨（TXFR1…）、其餘當股票"""
    code = code.strip().upper()
    C = api.Contracts
    tries = []
    if code[:3] in ("TSE", "OTC") and code[3:].isdigit():
        grp = lambda: getattr(C.Indexs, code[:3])  # noqa: E731
        tries += [lambda: grp()[code[3:]], lambda: grp()[code], lambda: getattr(grp(), code)]
    if not code.isdigit():
        tries += [lambda: C.Futures[code], lambda: C.Futures[code[:3]][code],
                  lambda: getattr(getattr(C.Futures, code[:3]), code)]
    tries += [lambda: C.Stocks[code]]
    for t in tries:
        try:
            c = t()
            if c is not None:
                return c
        except Exception:  # noqa: BLE001
            continue
    raise ValueError(f"找不到商品代號 {code}（股票如 2330；指數如 TSE001；期貨如 TXFR1）")


def is_futures(contract) -> bool:
    return "Future" in type(contract).__name__ or str(getattr(contract, "security_type", "")).endswith("FUT")


def fetch_minutes(api, code: str, start: dt.date, end: dt.date, day_session_only: bool = True,
                  chunk_days: int = 60, progress=None) -> pd.DataFrame:
    """下載 1 分 K（分段抓），回傳以「K 棒開始時間」為索引的 OHLC"""
    contract = resolve_contract(api, code)
    frames, cur = [], start
    spans = []
    while cur <= end:
        nxt = min(cur + dt.timedelta(days=chunk_days - 1), end)
        spans.append((cur, nxt))
        cur = nxt + dt.timedelta(days=1)
    for i, (a, b) in enumerate(spans):
        kb = api.kbars(contract=contract, start=a.isoformat(), end=b.isoformat())
        part = pd.DataFrame({k: list(getattr(kb, k)) for k in ("ts", "Open", "High", "Low", "Close")})
        if len(part):
            frames.append(part)
        if progress:
            progress((i + 1) / len(spans))
    if not frames:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    df = pd.concat(frames).drop_duplicates("ts")
    df.index = pd.to_datetime(df.pop("ts"))
    df = df.sort_index().astype(float)

    # Shioaji 的 ts 是 K 棒「結束」分鐘（第一根為 09:01 / 08:46）→ 改成開始時間，與其他資料一致
    first_min = df.index.to_series().groupby(df.index.date).min().dt.minute
    if len(first_min) and (first_min % 5 == 1).mean() > 0.5:
        df.index = df.index - pd.Timedelta(minutes=1)

    if day_session_only and is_futures(contract):
        t = df.index.time
        df = df[(t >= DAY_SESSION[0]) & (t < DAY_SESSION[1])]
    return df[["Open", "High", "Low", "Close"]]


def to_interval(m1: pd.DataFrame, interval: str) -> pd.DataFrame:
    """1 分 K → 5m/15m/30m/60m（每天從第一根起切）、1d、1wk"""
    if m1.empty:
        return m1
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if interval.endswith("m"):
        n = int(interval[:-1])
        return m1 if n == 1 else resample_ohlc(m1, n)
    daily = m1.groupby(m1.index.date).agg(agg)
    daily.index = pd.to_datetime(daily.index)
    if interval == "1d":
        return daily
    if interval == "1wk":
        return daily.resample("W-FRI").agg(agg).dropna()
    raise ValueError(f"不支援的週期 {interval}")


def usage_text(api) -> str:
    try:
        u = api.usage()
        used, limit = getattr(u, "bytes", None), getattr(u, "limit_bytes", None)
        if used is not None and limit:
            return f"今日流量 {used / 1e6:,.1f} / {limit / 1e6:,.0f} MB（{used / limit:.0%}）"
        return str(u)
    except Exception:  # noqa: BLE001
        return ""
