# -*- coding: utf-8 -*-
"""
台股代號清單（下拉選單用）與 Yahoo 後綴自動判斷：上市 → .TW，上櫃 → .TWO。

清單來源：twstock 套件內建的證交所／櫃買中心代號表（離線可用）。
要更新代號表：python -c "import twstock; twstock.__update_codes()"
"""
from __future__ import annotations

import re
from functools import lru_cache

KEEP_TYPES = {"股票", "ETF", "ETN", "特別股", "創新板", "臺灣存託憑證(TDR)"}
POPULAR = ["2330", "0050", "2454", "2317", "2308", "2382", "2303", "2881", "2882", "0056",
           "00878", "006208", "3711", "2412", "2891", "6488", "3105", "8069"]

YAHOO_INDEXES = [("^TWII", "加權指數"), ("^TWOII", "櫃買指數")]
SJ_SPECIAL = [("TSE001", "加權指數"), ("OTC101", "櫃買指數"),
              ("TXFR1", "台指期 近月"), ("TXFR2", "台指期 次月"), ("MXFR1", "小台指 近月")]


@lru_cache(maxsize=1)
def stock_table() -> dict[str, dict]:
    """{代號: {name, market, suffix, group, type}}，只留上市（含創新板）與上櫃的股票／ETF 類"""
    try:
        import twstock
        codes = twstock.codes
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for code, v in codes.items():
        if v.type not in KEEP_TYPES:
            continue
        if v.market.startswith("上市"):
            suffix = ".TW"
        elif v.market == "上櫃":
            suffix = ".TWO"
        else:
            continue
        out[code] = {"name": v.name, "market": "上市" if suffix == ".TW" else "上櫃",
                     "suffix": suffix, "group": v.group or v.type, "type": v.type}
    return out


def _label(code: str, info: dict) -> str:
    return f"{code} {info['name']}（{info['market']}・{info['group']}）"


def options(source: str) -> list[str]:
    """下拉選單選項：指數／期貨在前，熱門股次之，其餘依代號排序"""
    tbl = stock_table()
    special = YAHOO_INDEXES if source == "yahoo" else SJ_SPECIAL
    head = [f"{c} {n}" for c, n in special]
    pop = [_label(c, tbl[c]) for c in POPULAR if c in tbl]
    rest = [_label(c, tbl[c]) for c in sorted(tbl) if c not in POPULAR]
    return head + pop + rest


def code_of(choice: str) -> str:
    """'2330 台積電（上市・半導體業）' → '2330'；使用者自行輸入的也接受（如 '2330.tw'、'6488'）"""
    return choice.strip().split()[0].upper() if choice and choice.strip() else ""


def yahoo_candidates(choice: str) -> list[str]:
    """回傳要嘗試的 Yahoo 代號：已知上市／上櫃直接給後綴；未知的先試 .TW 再試 .TWO"""
    code = code_of(choice)
    if not code:
        return []
    if code.startswith("^") or re.search(r"\.[A-Z]+$", code):
        return [code]
    info = stock_table().get(code)
    if info:
        return [code + info["suffix"]]
    return [code + ".TW", code + ".TWO"]


def display_name(choice: str) -> str:
    code = code_of(choice).split(".")[0]
    info = stock_table().get(code)
    if info:
        return f"{code} {info['name']}"
    for c, n in YAHOO_INDEXES + SJ_SPECIAL:
        if c == code:
            return f"{c} {n}"
    return code
