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
    wait_contracts(api)
    return api, accounts


def wait_contracts(api, timeout: float = 60.0) -> None:
    """確認合約表已下載完成（沒下載完時查代號會拿到空的或錯的物件）"""
    import time

    def status():
        return str(getattr(api.Contracts, "status", "")).split(".")[-1]

    if status() != "Fetched":
        try:
            api.fetch_contracts(contract_download=True)
        except Exception:  # noqa: BLE001
            pass
    t0 = time.time()
    while status() not in ("Fetched", "") and time.time() - t0 < timeout:
        time.sleep(0.5)


def account_lines(accounts) -> list[str]:
    """帳號只顯示類別與末 4 碼"""
    out = []
    for a in accounts or []:
        kind = type(a).__name__.replace("Account", "") or "帳戶"
        aid = str(getattr(a, "account_id", ""))
        out.append(f"{kind}　***{aid[-4:]}" if aid else kind)
    return out


def _is_contract(obj) -> bool:
    """只接受單一合約（Stock/Future/Option/Index…），排除合約群組等其他物件"""
    if obj is None:
        return False
    try:
        import shioaji as sj
        kinds = tuple(t for t in (getattr(sj, n, None) for n in
                      ("BaseContract", "Contract", "Stock", "Future", "Option", "Index", "ComboContract",
                                "StockInfo", "IndexInfo", "FuturesInfo", "OptionInfo", "WarrantInfo"))
                      if isinstance(t, type))
        if kinds and isinstance(obj, kinds):
            return True
    except Exception:  # noqa: BLE001
        pass
    return bool(getattr(obj, "code", None)) and bool(getattr(obj, "security_type", None))


def _scan_index(C, exch: str, num: str):
    """逐一掃描指數合約表，比對 交易所＋代號 或 symbol（如 TSE001）"""
    try:
        for grp in C.Indexs:
            for c in grp:
                ex = str(getattr(c, "exchange", "")).split(".")[-1].upper()
                cd = str(getattr(c, "code", ""))
                sym = str(getattr(c, "symbol", "") or "").upper()
                if sym == exch + num or (cd == num and exch in ex):
                    return c
    except Exception:  # noqa: BLE001
        pass
    return None


def resolve_contract(api, code: str):
    """依代號找合約：指數 TSE/OTC 開頭、期貨（TXFR1…）、其餘當股票／ETF"""
    code = code.strip().upper().split(".")[0]          # 2330.TW → 2330
    wait_contracts(api, timeout=30)
    C = api.Contracts
    tries = []
    if code[:3] in ("TSE", "OTC") and code[3:].isdigit():
        tries += [lambda: getattr(C.Indexs, code[:3])[code[3:]],
                  lambda: getattr(getattr(C.Indexs, code[:3]), code),
                  lambda: getattr(C.Indexs, code[:3])[code],
                  lambda: C.Indexs[code[3:]], lambda: C.Indexs[code]]
    tries += [lambda: C.Stocks[code]]                    # 股票、ETF（含 00631L 這類英文字尾）
    if not code.isdigit():
        tries += [lambda: C.Futures[code], lambda: getattr(C.Futures, code[:3])[code]]
    for t in tries:
        try:
            c = t()
        except Exception:  # noqa: BLE001
            continue
        if _is_contract(c):
            return c
    if code[:3] in ("TSE", "OTC") and code[3:].isdigit():
        c = _scan_index(C, code[:3], code[3:])
        if c is not None:
            return c
        try:   # 合約表裡找不到時，直接指定「交易所＋代號」（kbars 接受 BaseContract）
            import shioaji as sj
            return sj.BaseContract(security_type="IND", exchange=code[:3], code=code[3:])
        except Exception:  # noqa: BLE001
            pass
    try:   # 股票／ETF：用代號表判斷上市（TSE）或上櫃（OTC）後直接指定
        import shioaji as sj
        from tw_symbols import stock_table
        info = stock_table().get(code)
        if info:
            return sj.BaseContract(security_type="STK", exchange="TSE" if info["suffix"] == ".TW" else "OTC",
                                   code=code)
    except Exception:  # noqa: BLE001
        pass
    st = str(getattr(C, "status", "")).split(".")[-1]
    hint = "（合約表尚未下載完成，請稍後再按一次讀取，或登出後重新登入）" if st and st != "Fetched" else ""
    raise ValueError(f"找不到商品代號 {code}{hint}。股票如 2330；指數如 TSE001；期貨如 TXFR1")


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
