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

# 常用清單（大致依市值／成交熱度排列；不在代號表中的會自動略過）
POPULAR_TWSE = (
    "2330 2317 2454 2308 2382 2412 2881 2882 2891 2303 3711 2886 2884 1216 2357 3231 2345 6669 "
    "3008 2885 2892 2880 5880 2883 2887 2890 1301 1303 1326 6505 2002 2603 2609 2615 2207 2395 "
    "3045 4904 2912 1101 3034 2379 3037 2327 4938 2301 3017 2360 3661 3443 2376 2356 2324 2353 "
    "6415 2408 2344 3481 2409 2618 2610 1590 2059 3653 2368 2383 6446 1519 1513 1503 2049 9910 "
    "2105 1402 2801 5871 2888 2834 3533 2474 1476 8046 3044 6239 2449 2404 3714 4958 5269 6409 "
    "3406 2385 2492 6285 8210 3665 2313 2337 3035 6789 1605 2371"
).split()
POPULAR_TPEX = (
    "6488 3105 8069 5347 6147 3529 5274 8299 3293 6274 4966 3324 5483 6121 3211 8086 4743 6180 "
    "5371 3374 6223 4979 3363 6683 3260 8436 6643 3141 6510 5289 4162 3264 6182 1785 8358 4123 "
    "6548 3680 3227 8044"
).split()
POPULAR_ETF = (
    "0050 0056 00878 006208 00919 00929 00940 00713 00692 00881 00891 00892 00631L 00632R "
    "00679B 00687B 00937B 00720B 00757 00830 00662 0052 00900 00915 00918 00646 00927 00934 "
    "00939 00946"
).split()
POPULAR = list(dict.fromkeys(POPULAR_TWSE + POPULAR_TPEX + POPULAR_ETF))

CAT_ALL = "全部（常用在前）"
CAT_TWSE = "⭐ 常用上市個股"
CAT_TPEX = "⭐ 常用上櫃個股"
CAT_ETF = "⭐ 常用 ETF"
CAT_SPECIAL = {"yahoo": "指數", "sj": "指數／期貨"}
CAT_ALL_ETF = "所有 ETF／ETN"
INDUSTRY_PREFIX = "產業｜"

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


def categories(source: str) -> list[str]:
    """分類下拉：全部、常用三組、指數（期貨）、所有 ETF、各產業"""
    tbl = stock_table()
    groups = sorted({v["group"] for v in tbl.values() if v["type"] in ("股票", "創新板")})
    return ([CAT_ALL, CAT_TWSE, CAT_TPEX, CAT_ETF, CAT_SPECIAL[source], CAT_ALL_ETF]
            + [INDUSTRY_PREFIX + g for g in groups])


def options(source: str, category: str = CAT_ALL) -> list[str]:
    """下拉選單選項。全部：指數／期貨在前，常用（上市→上櫃→ETF）次之，其餘依代號排序"""
    tbl = stock_table()
    special = [f"{c} {n}" for c, n in (YAHOO_INDEXES if source == "yahoo" else SJ_SPECIAL)]
    pick = lambda codes: [_label(c, tbl[c]) for c in codes if c in tbl]  # noqa: E731
    if category == CAT_TWSE:
        return pick(POPULAR_TWSE)
    if category == CAT_TPEX:
        return pick(POPULAR_TPEX)
    if category == CAT_ETF:
        return pick(POPULAR_ETF)
    if category == CAT_SPECIAL.get(source):
        return special
    if category == CAT_ALL_ETF:
        return pick(sorted(c for c, v in tbl.items() if v["type"] in ("ETF", "ETN")))
    if category.startswith(INDUSTRY_PREFIX):
        g = category[len(INDUSTRY_PREFIX):]
        codes = [c for c, v in tbl.items() if v["group"] == g and v["type"] in ("股票", "創新板")]
        pop_rank = {c: i for i, c in enumerate(POPULAR)}
        return pick(sorted(codes, key=lambda c: (pop_rank.get(c, 10**6), c)))   # 常用的排前面
    return special + pick(POPULAR) + pick(sorted(c for c in tbl if c not in set(POPULAR)))


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
