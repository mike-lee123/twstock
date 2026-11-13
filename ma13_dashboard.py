# -*- coding: utf-8 -*-
"""
13MA 的魔力 —— Streamlit 儀表板

啟動：
    streamlit run ma13_dashboard.py

規則與計算全部沿用 ma13_system.py（指標、費氏、開盤劇情、回測），
本檔只負責資料讀取、互動參數與視覺化。
教學用途，成本可在側欄設定，不構成投資建議。
"""
from __future__ import annotations

import datetime as dt
import hmac
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ma13_system import (
    HALF_RETRACE,
    MA_LEN,
    OPEN_ENERGY_MULT,
    add_indicators,
    backtest,
    backtest_mtf,
    bt_stats,
    fib_retracements,
    fib_sequence_levels,
    OVERNIGHT_EXITS,
    OVERNIGHT_FILTERS,
    odd_lot_cost_pct,
    opening_reports,
    overnight_returns,
    overnight_stats,
    overnight_table,
    resample_ohlc,
)
import ma13_closed_loop as cl
import sinopac_data as sjd
import tw_symbols as tws

BIAS_OPTS = {"strict": "三條件齊（13MA＋K 50＋MACD 柱 0）", "ma": "只看 30 分K 在 13MA 之上／下"}
EXIT_OPTS = {"lo": "5 分K 反向穿越 13MA（較快）", "hi": "30 分K 反向穿越 13MA（較慢）"}

UP, DOWN = "#e5484d", "#2f9e44"          # 台股慣例：紅漲綠跌
MA_COLOR, K_COLOR, D_COLOR = "#f59f00", "#1c7ed6", "#ae3ec9"

st.set_page_config(page_title="13MA 的魔力", page_icon="📈", layout="wide")


# ───────────────────────── 密碼保護（部署到雲端時用） ─────────────────────────
def get_secret(name: str) -> str:
    """先讀 Streamlit secrets（雲端在 App settings → Secrets 設定），再讀環境變數"""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # noqa: BLE001 — 本機沒有 secrets.toml 時會丟例外
        pass
    return os.environ.get(name, "")


APP_PASSWORD = get_secret("APP_PASSWORD")
if APP_PASSWORD and not st.session_state.get("authed"):
    st.title("📈 13MA 的魔力")
    with st.form("app_login"):
        pw = st.text_input("請輸入密碼", type="password")
        if st.form_submit_button("進入", type="primary"):
            if hmac.compare_digest(pw, APP_PASSWORD):
                st.session_state["authed"] = True
                st.rerun()
            st.error("密碼錯誤。")
    st.stop()

# 只有設了 APP_PASSWORD（且已通過）才自動帶入永豐金鑰，避免公開網址把金鑰送給任何訪客
PREFILL_KEYS = bool(APP_PASSWORD)
TODAY = pd.Timestamp.now(tz="Asia/Taipei").date()   # 雲端主機是 UTC，日期一律用台北時間


# ───────────────────────── 資料 ─────────────────────────
def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("Asia/Taipei").tz_localize(None)
    return df[["Open", "High", "Low", "Close"]].dropna()


@st.cache_data(ttl=600, show_spinner="下載行情中…")
def fetch_yf(symbol: str, period: str, interval: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(symbol, period=period, interval=interval,
                     auto_adjust=False, progress=False)
    return _clean(df)


@st.cache_data(ttl=86400, show_spinner="判斷上市／上櫃…")
def resolve_yahoo(choice: str) -> str:
    """清單內的代號直接加 .TW／.TWO；清單外的依序試 .TW、.TWO，哪個抓得到資料就用哪個"""
    cands = tws.yahoo_candidates(choice)
    if len(cands) <= 1:
        return cands[0] if cands else ""
    for s in cands:
        try:
            if not fetch_yf(s, "5d", "1d").empty:
                return s
        except Exception:  # noqa: BLE001
            continue
    return cands[0]


@st.cache_data(show_spinner=False)
def read_csv(raw: bytes) -> pd.DataFrame:
    from io import BytesIO
    df = pd.read_csv(BytesIO(raw), parse_dates=[0], index_col=0)
    return _clean(df)


@st.cache_data(ttl=600, show_spinner="向永豐 API 下載 1 分K…（區間長會花一點時間）")
def fetch_sj(_api, session_id: str, code: str, start: dt.date, end: dt.date,
             day_only: bool) -> pd.DataFrame:
    """_api 不參與快取鍵；session_id 讓換帳號／重新登入後重新抓"""
    return sjd.fetch_minutes(_api, code, start, end, day_session_only=day_only)


def sinopac_login_panel():
    """側欄：永豐 API 填空登入；已登入則顯示狀態與登出"""
    sess = st.session_state.get("sj")
    if sess is None:
        with st.form("sj_login"):
            st.markdown("**🔑 永豐 API 登入**（只讀行情，不下單）")
            api_key = st.text_input("API Key", value=get_secret("SJ_API_KEY") if PREFILL_KEYS else "",
                                    placeholder="貼上 API Key")
            secret = st.text_input("Secret Key", value=get_secret("SJ_SEC_KEY") if PREFILL_KEYS else "",
                                   type="password", placeholder="貼上 Secret Key")
            sim = st.toggle("模擬環境", value=False, help="新申請、尚未通過測試的金鑰請先用模擬環境")
            submitted = st.form_submit_button("登入", type="primary")
            st.caption("金鑰只存在這次瀏覽階段的記憶體，不寫入檔案。"
                       "設了 APP_PASSWORD 時，可在 secrets 設 SJ_API_KEY／SJ_SEC_KEY 自動帶入。")
        if submitted:
            if not api_key or not secret:
                st.error("請填 API Key 與 Secret Key。")
            else:
                try:
                    with st.spinner("登入永豐 API…"):
                        api, accounts = sjd.login(api_key, secret, simulation=sim)
                    st.session_state["sj"] = {
                        "api": api, "sim": sim, "accounts": sjd.account_lines(accounts),
                        "id": f"{dt.datetime.now():%H%M%S%f}",
                    }
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"登入失敗：{e}")
        return None
    st.success(f"已登入永豐 API（{'模擬' if sess['sim'] else '正式'}環境）", icon="✅")
    if sess["accounts"]:
        st.caption("帳戶：" + "、".join(sess["accounts"]))
    usage = sjd.usage_text(sess["api"])
    if usage:
        st.caption(usage)
    if st.button("登出"):
        try:
            sess["api"].logout()
        except Exception:  # noqa: BLE001
            pass
        del st.session_state["sj"]
        st.rerun()
    return sess


# ───────────────────────── 側欄參數 ─────────────────────────
with st.sidebar:
    st.header("⚙️ 參數")
    source = st.radio("資料來源", ["Yahoo Finance", "永豐 Shioaji", "上傳 CSV"], horizontal=True)
    sj_sess = None
    if source == "永豐 Shioaji":
        sj_sess = sinopac_login_panel()
        sj_cat = st.selectbox("代號分類", tws.categories("sj"), key="cat_sj",
                              help="先選分類縮小清單；「全部」可搜尋所有上市櫃股票")
        sj_opts = tws.options("sj", sj_cat)
        sym_choice = st.selectbox(
            "商品代號", sj_opts, index=next((i for i, o in enumerate(sj_opts) if o.startswith("2330 ")), 0),
            key=f"sym_sj_{sj_cat}", accept_new_options=True,
            help="可直接打代號或名稱搜尋；清單外的代號也可直接輸入後按 Enter。"
                 "指數 TSE001／OTC101、期貨 TXFR1／MXFR1 在最上面。")
        symbol = tws.code_of(sym_choice)
        interval = st.selectbox("K 棒週期", ["5m", "15m", "30m", "60m", "1d", "1wk"], index=2)
        today = TODAY
        d1, d2 = st.columns(2)
        sj_start = d1.date_input("起", today - dt.timedelta(days=180), max_value=today)
        sj_end = d2.date_input("迄", today, max_value=today)
        sj_day_only = st.toggle("期貨只取日盤", value=True, help="夜盤會打亂開盤劇情與隔日沖的「日」")
        period, upload = f"{sj_start}～{sj_end}", None
        st.caption("永豐 1 分K 可往前抓數年，分K 不受 Yahoo 60 天限制；區間越長下載越久、流量越大。")
    elif source == "Yahoo Finance":
        yf_cat = st.selectbox("代號分類", tws.categories("yahoo"), key="cat_yahoo",
                              help="先選分類縮小清單；「全部」可搜尋所有上市櫃股票")
        sym_choice = st.selectbox(
            "代號", tws.options("yahoo", yf_cat), index=0, key=f"sym_yahoo_{yf_cat}",
            accept_new_options=True,
            help="可直接打代號或名稱搜尋（如 2330、台積電）；自動判斷上市 .TW／上櫃 .TWO。"
                 "清單外的代號也可直接輸入後按 Enter。")
        symbol = resolve_yahoo(sym_choice)
        if symbol and not symbol.startswith("^"):
            mkt = "上櫃" if symbol.endswith(".TWO") else ("上市" if symbol.endswith(".TW") else "")
            st.caption(f"Yahoo 代號：**{symbol}**" + (f"（{mkt}）" if mkt else ""))
        interval = st.selectbox("K 棒週期", ["5m", "15m", "30m", "60m", "1d", "1wk"], index=2)
        default_period = "60d" if interval.endswith("m") else "2y"
        period = st.text_input("資料期間", default_period,
                               help="分 K 最多 60 天（如 30d、60d）；日 K 可用 1y、2y、5y")
        upload = None
    else:
        upload = st.file_uploader("CSV（欄位 Datetime,Open,High,Low,Close）", type="csv")
        interval = st.selectbox("CSV 的 K 棒週期", ["5m", "15m", "30m", "60m", "1d"], index=0,
                                help="選 5m 時才會產生開盤劇情")
        symbol, period = (upload.name if upload else "CSV"), ""

    st.divider()
    st.subheader("回測")
    bt_mode = st.radio("回測方式", ["單一週期", "30分K定方向＋5分K進場"],
                       help="單一週期：用上面選的 K 棒週期。組合：一律用 5 分K 合成 30 分K"
                            "（Yahoo 最近 60 天；永豐用上面的日期區間），與上面的週期設定無關。")
    if bt_mode != "單一週期":
        bias_mode = st.selectbox("30 分K 方向判斷", list(BIAS_OPTS), format_func=BIAS_OPTS.get)
        exit_mode = st.selectbox("出場依據", list(EXIT_OPTS), format_func=EXIT_OPTS.get)
    cost = st.number_input("每筆來回成本（點）", min_value=0.0, value=0.0, step=1.0,
                           help="手續費＋稅＋滑價折成點數，每筆交易扣一次")
    long_only = st.toggle("只做多（個股）", value=False)
    use_stop = st.toggle("固定停損", value=False)
    stop = st.number_input("停損點數", min_value=0.0, value=100.0, step=10.0,
                           disabled=not use_stop) if use_stop else None
    min_profit = st.number_input("回吐一半的最低獲利門檻（點）", min_value=0.0, value=0.0, step=5.0,
                                 help="最高獲利超過此值才啟動「回吐一半就砍」")

    st.divider()
    st.subheader("顯示")
    show_bars = st.slider("圖上顯示最近幾根 K 棒", 60, 1000, 300, step=20)
    fib_lookback = st.slider("費氏區間（最近幾根）", 20, 300, 60, step=10)
    show_fib = st.toggle("圖上畫費氏關卡", value=True)
    show_trades = st.toggle("圖上標示進出場", value=True)
    st.caption("教學用規則化程式，未計手續費、滑價與稅，不構成投資建議。")


# ───────────────────────── 載入 ─────────────────────────
sj_m1 = None   # 永豐 1 分K（所有週期都由它合成）
if source != "上傳 CSV" and not (symbol or "").strip():
    st.title("📈 13MA 的魔力")
    st.info("請在左側「代號」選擇或輸入商品代號（例如 2330），輸入後按 Enter。", icon="👈")
    st.stop()
try:
    if source == "上傳 CSV":
        if upload is None:
            st.title("📈 13MA 的魔力")
            st.info("請在左側上傳 CSV。")
            st.stop()
        raw_df = read_csv(upload.getvalue())
    elif source == "永豐 Shioaji":
        if sj_sess is None:
            st.title("📈 13MA 的魔力")
            st.info("請在左側填入永豐 API Key 與 Secret Key 後按「登入」。", icon="🔑")
            st.stop()
        if sj_start > sj_end:
            st.error("起始日不能晚於結束日。")
            st.stop()
        sj_m1 = fetch_sj(sj_sess["api"], sj_sess["id"], symbol.strip().upper(), sj_start, sj_end, sj_day_only)
        raw_df = sjd.to_interval(sj_m1, interval)
    else:
        raw_df = fetch_yf(symbol, period, interval)
except Exception as e:  # noqa: BLE001
    st.error(f"讀取資料失敗：{e}")
    st.stop()

if raw_df.empty or len(raw_df) < MA_LEN + 2:
    st.error("抓不到足夠資料，請確認代號、週期（Yahoo 分 K 最多 60 天）、永豐日期區間或 CSV 檔。")
    st.stop()

df = add_indicators(raw_df)

# 開盤劇情與多週期回測一定用 5 分 K
intraday = None
if source == "上傳 CSV":
    if interval == "5m":
        intraday = raw_df
elif sj_m1 is not None:
    intraday = sjd.to_interval(sj_m1, "5m")
else:
    try:
        intraday = fetch_yf(symbol, "60d", "5m")
    except Exception:  # noqa: BLE001
        intraday = None
has_intraday = intraday is not None and len(intraday) > 30 * 6
reps = opening_reports(intraday) if has_intraday else []

bt_kw = dict(long_only=long_only, stop=stop, min_profit_for_half=min_profit, cost=cost)
d5 = d30 = None
if has_intraday:
    d5 = add_indicators(intraday)
    d30 = add_indicators(resample_ohlc(intraday, 30))

if bt_mode == "單一週期":
    bt = backtest(df, **bt_kw)
elif d5 is None:
    st.error("組合回測需要 5 分 K 資料（Yahoo 最近 60 天、永豐 API，或上傳 5m 的 CSV）。")
    st.stop()
else:
    bt = backtest_mtf(d5, d30, bias_mode=bias_mode, exit_mode=exit_mode, **bt_kw)


# ───────────────────────── 標題與目前狀態 ─────────────────────────
last, prev = df.iloc[-1], df.iloc[-2]
name_label = tws.display_name(symbol) if source != "上傳 CSV" else symbol
if source == "Yahoo Finance" and not symbol.startswith("^"):
    name_label += f"（{symbol}）"
src_label = f"{name_label}　{interval}" + (f" / {period}" if period else "")
st.title("📈 13MA 的魔力")
st.caption(f"資料：{src_label}　共 {len(df):,} 根 K 棒　最新 {df.index[-1]}")

is_bull = last["Close"] > last["MA13"] and last["K"] > 50 and last["OSC"] > 0
below = last["Close"] < last["MA13"]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("收盤", f"{last['Close']:,.2f}", f"{last['Close'] - prev['Close']:+,.2f}",
          delta_color="inverse")  # inverse → 漲紅跌綠
c2.metric("13MA", f"{last['MA13']:,.2f}", f"距離 {last['Close'] - last['MA13']:+,.2f}",
          delta_color="inverse")
c3.metric("K / D", f"{last['K']:.1f} / {last['D']:.1f}")
c4.metric("MACD 柱", f"{last['OSC']:+.2f}")
c5.metric("多空開關", "多頭 🔴" if is_bull else ("偏空 🟢" if below else "條件不齊 ⚪"))

if is_bull:
    st.error("**多頭**：站上 13MA、KD>50、MACD 過 0 → 續抱多單，跌破 13MA 再說", icon="🔴")
elif below:
    st.success("**在 13MA 之下**：偏空，等三個條件一起站回才做多", icon="🟢")
else:
    st.warning("**站上 13MA 但 KD／MACD 尚未同步**：條件不齊，不追", icon="⚪")
if last["過兩根高"]:
    st.info("本根過前兩根高：開始動（三根法則）", icon="⬆️")
if last["破兩根低"]:
    st.info("本根破前兩根低：開始往下跑（三根法則）", icon="⬇️")
if last["飆股標記"]:
    st.info(f"KD>80 且低點沒破：飆股標記，目標價 ×3 ≈ {last['Close'] * 3:,.2f}"
            "（需天天漲停的極端條件）", icon="🚀")

recent = df.tail(fib_lookback)
fib_lo, fib_hi = recent["Low"].min(), recent["High"].max()
fibs = fib_retracements(fib_lo, fib_hi)


# ───────────────────────── 圖表 ─────────────────────────
def build_chart(view: pd.DataFrame) -> go.Figure:
    x = view.index.strftime("%Y-%m-%d %H:%M" if interval.endswith("m") else "%Y-%m-%d")
    pos = {ts: lbl for ts, lbl in zip(view.index, x)}
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.6, 0.2, 0.2])
    fig.add_trace(go.Candlestick(
        x=x, open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"],
        increasing=dict(line=dict(color=UP), fillcolor=UP),
        decreasing=dict(line=dict(color=DOWN), fillcolor=DOWN), name="K 線"), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=view["MA13"], name="13MA",
                             line=dict(color=MA_COLOR, width=2)), 1, 1)

    lo, hi = view["Low"].min(), view["High"].max()
    y_top, y_bot = hi, lo
    if show_fib:
        # 只畫落在畫面價格區間附近的關卡，避免遠方的延伸線把 K 線壓扁
        pad = (hi - lo) * 0.15
        for name, lv in fibs.items():
            if not (lo - pad <= lv <= hi + pad):
                continue
            y_top, y_bot = max(y_top, lv), min(y_bot, lv)
            fig.add_hline(y=lv, line=dict(color="#868e96", dash="dot", width=1),
                          annotation_text=f"{name.split('（')[0]} {lv:,.0f}", annotation_position="right",
                          annotation_font_size=10, row=1, col=1)

    if show_trades and bt.trades:
        # 交易時間對到「包含它的那根 K 棒」（組合回測是 5 分 K 時間，圖可能是 30 分或日 K）
        def bar_of(t):
            i = view.index.searchsorted(t, side="right") - 1
            return x[i] if i >= 0 and t <= view.index[-1] + (view.index[-1] - view.index[-2]) else None
        pos = {t2: bar_of(t2) for t in bt.trades for t2 in (t.entry_time, t.exit_time)}
        pos = {k: v for k, v in pos.items() if v is not None}
        tr = [t for t in bt.trades if t.entry_time in pos or t.exit_time in pos]
        for side, sym, color in (("多", "triangle-up", UP), ("空", "triangle-down", DOWN)):
            ent = [t for t in tr if t.side == side and t.entry_time in pos]
            if ent:
                fig.add_trace(go.Scatter(
                    x=[pos[t.entry_time] for t in ent], y=[t.entry for t in ent], mode="markers",
                    marker=dict(symbol=sym, size=12, color=color, line=dict(width=1, color="white")),
                    name=f"{side}單進場",
                    hovertext=[f"{side}單進場 {t.entry:,.2f}" for t in ent], hoverinfo="text"), 1, 1)
        ex = [t for t in tr if t.exit_time in pos]
        if ex:
            fig.add_trace(go.Scatter(
                x=[pos[t.exit_time] for t in ex], y=[t.exit for t in ex], mode="markers",
                marker=dict(symbol="x", size=10, color="#495057"), name="出場",
                hovertext=[f"出場 {t.exit:,.2f}｜{t.reason}｜損益 {t.pnl:+,.1f}" for t in ex],
                hoverinfo="text"), 1, 1)

    fig.add_trace(go.Scatter(x=x, y=view["K"], name="K", line=dict(color=K_COLOR),
                             legend="legend2"), 2, 1)
    fig.add_trace(go.Scatter(x=x, y=view["D"], name="D", line=dict(color=D_COLOR),
                             legend="legend2"), 2, 1)
    for y, c in ((50, "#868e96"), (80, UP), (20, DOWN)):
        fig.add_hline(y=y, line=dict(color=c, dash="dot", width=1), row=2, col=1)

    fig.add_trace(go.Bar(x=x, y=view["OSC"], name="MACD 柱", legend="legend3",
                         marker_color=np.where(view["OSC"] >= 0, UP, DOWN)), 3, 1)
    fig.add_trace(go.Scatter(x=x, y=view["DIF"], name="DIF", legend="legend3",
                             line=dict(color=K_COLOR, width=1)), 3, 1)
    fig.add_trace(go.Scatter(x=x, y=view["MACD"], name="MACD", legend="legend3",
                             line=dict(color=MA_COLOR, width=1)), 3, 1)

    # 每個面板各自一組圖例，放在面板左上角內側；面板名稱放在 Y 軸標題，不再跟圖例搶位置
    def panel_legend(row: int, title: str) -> dict:
        top = fig.layout[f"yaxis{'' if row == 1 else row}"].domain[1]
        return dict(title=dict(text=f"<b>{title}</b>　", font=dict(size=12)), orientation="h",
                    x=0.005, xanchor="left", y=top - 0.005, yanchor="top", font=dict(size=11),
                    bgcolor="rgba(255,255,255,0.75)", bordercolor="rgba(0,0,0,0.08)", borderwidth=1)

    fig.update_xaxes(type="category", nticks=10, showgrid=False)
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    for row, ttl in ((1, "價格"), (2, "KD"), (3, "MACD")):
        fig.update_yaxes(title_text=ttl, title_font=dict(size=11), automargin=True, row=row, col=1)
    fig.update_layout(
        height=820, margin=dict(l=10, r=120, t=20, b=10), hovermode="x unified", bargap=0.1,
        legend=panel_legend(1, "K 線 + 13MA"),
        legend2=panel_legend(2, "KD(9,3,3)"),
        legend3=panel_legend(3, "MACD(12,26,9)"),
    )
    fig.update_xaxes(automargin=True)
    span = y_top - y_bot
    fig.update_yaxes(range=[y_bot - span * 0.04, y_top + span * 0.14], row=1, col=1)  # 上方留白給圖例
    fig.update_yaxes(range=[-5, 125], tickvals=[20, 50, 80], row=2, col=1)
    m_lo = float(np.nanmin(view[["OSC", "DIF", "MACD"]].to_numpy()))
    m_hi = float(np.nanmax(view[["OSC", "DIF", "MACD"]].to_numpy()))
    m_span = (m_hi - m_lo) or 1.0
    fig.update_yaxes(range=[m_lo - m_span * 0.05, m_hi + m_span * 0.35], row=3, col=1)
    return fig


tab_chart, tab_open, tab_bt, tab_loop, tab_odd, tab_fib, tab_rules = st.tabs(
    ["📈 走勢圖", "🌅 開盤劇情", "📊 回測", "🔁 閉環系統", "🌙 零股隔日沖", "📐 費氏座標", "📖 規則"])

with tab_chart:
    st.plotly_chart(build_chart(df.tail(show_bars)))
    with st.expander("最近訊號"):
        sig_cols = ["多頭成立", "空頭成立", "站上13MA", "跌破13MA", "過兩根高", "破兩根低", "飆股標記"]
        sig = df[sig_cols].tail(show_bars)
        rows = [{"時間": ts, "訊號": "、".join(c for c in sig_cols if r[c]), "收盤": round(df.at[ts, "Close"], 2)}
                for ts, r in sig.iterrows() if r.any()]
        st.dataframe(pd.DataFrame(rows[::-1]), hide_index=True, width="stretch")


# ───────────────────────── 開盤劇情 ─────────────────────────
with tab_open:
    if not reps:
        st.info("沒有 5 分 K 資料，無法產生開盤劇情（CSV 需選 5m 週期；"
                "Yahoo 5 分 K 最多 60 天）。")
    else:
        rep_df = pd.DataFrame([{
            "日期": r.date, "昨收": r.prev_close, "開盤": r.open, "跳空": r.gap,
            "跳空%": r.gap_pct, "第一根振幅": r.first_bar_range,
            f"預估幅度(×{OPEN_ENERGY_MULT})": r.projected_range,
            "實際幅度": r.day_high - r.day_low,
            "缺口性質": ("大缺口・看力量→出貨" if abs(r.gap) > r.first_bar_range else "小缺口・看方向→進貨"),
            "缺口一半": r.half_fill_level,
            "回補": "全補" if r.full_filled else ("補一半" if r.half_filled else "未補"),
            "高": r.day_high, "低": r.day_low, "收": r.close,
        } for r in reps])
        hit = (rep_df["實際幅度"] <= rep_df[f"預估幅度(×{OPEN_ENERGY_MULT})"]).mean()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("統計天數", f"{len(reps)} 天")
        m2.metric(f"第一根 ×{OPEN_ENERGY_MULT} 涵蓋實際幅度", f"{hit:.0%}")
        m3.metric("缺口全補比例", f"{(rep_df['回補'] == '全補').mean():.0%}")
        m4.metric("平均跳空", f"{rep_df['跳空'].abs().mean():,.1f} 點")

        r = reps[-1]
        st.subheader(f"最新一天：{r.date}")
        st.code(r.text(), language=None)

        fig2 = go.Figure()
        fig2.add_bar(x=rep_df["日期"].astype(str), y=rep_df["實際幅度"], name="實際幅度",
                     marker_color="#adb5bd")
        fig2.add_scatter(x=rep_df["日期"].astype(str), y=rep_df[f"預估幅度(×{OPEN_ENERGY_MULT})"],
                         name=f"第一根 5 分 K ×{OPEN_ENERGY_MULT}", mode="lines+markers",
                         line=dict(color=MA_COLOR))
        fig2.add_bar(x=rep_df["日期"].astype(str), y=rep_df["跳空"], name="跳空點數",
                     marker_color=np.where(rep_df["跳空"] >= 0, UP, DOWN))
        fig2.update_layout(height=380, barmode="overlay", hovermode="x unified",
                           margin=dict(l=10, r=10, t=30, b=10),
                           legend=dict(orientation="h", y=1.1), xaxis_type="category")
        st.plotly_chart(fig2)

        st.dataframe(
            rep_df.iloc[::-1], hide_index=True, width="stretch",
            column_config={c: st.column_config.NumberColumn(format="%.0f") for c in
                           ["昨收", "開盤", "跳空", "第一根振幅", f"預估幅度(×{OPEN_ENERGY_MULT})",
                            "實際幅度", "缺口一半", "高", "低", "收"]}
            | {"跳空%": st.column_config.NumberColumn(format="%+.2f%%")})


# ───────────────────────── 回測 ─────────────────────────
with tab_bt:
    cost_txt = f"每筆扣成本 {cost:g} 點" if cost else "未計成本"
    stop_txt = f"、固定停損 {stop:g} 點" if stop else ""
    if bt_mode == "單一週期":
        st.caption(f"**單一週期（{interval}）**　進場：多頭／空頭成立（13MA＋KD 50＋MACD 柱 0 同步）；"
                   f"出場：反向穿越 13MA、獲利回吐超過 {HALF_RETRACE:.0%}{stop_txt}。以收盤價成交，{cost_txt}。")
    else:
        st.caption(f"**30分K定方向＋5分K進場**　方向：{BIAS_OPTS[bias_mode]}（只用已收盤的 30 分K）；"
                   f"進場：方向一致且 5 分K 三條件齊；出場：{EXIT_OPTS[exit_mode]}、30 分K 方向翻轉、"
                   f"獲利回吐超過 {HALF_RETRACE:.0%}{stop_txt}。以 5 分K 收盤價成交，{cost_txt}。")

    if d5 is not None:
        with st.expander("📋 方法比較：5 分K 單獨 vs 30 分K 單獨 vs 組合（同一份 5 分K 資料、同樣成本）",
                         expanded=bt_mode != "單一週期"):
            runs = {
                "5 分K 單獨": backtest(d5, **bt_kw),
                "30 分K 單獨": backtest(d30, **bt_kw),
            }
            for b in BIAS_OPTS:
                for e in EXIT_OPTS:
                    runs[f"組合｜方向：{'三條件' if b == 'strict' else '只看13MA'}｜出場：{e.replace('lo', '5分K').replace('hi', '30分K')}"] = \
                        backtest_mtf(d5, d30, bias_mode=b, exit_mode=e, **bt_kw)
            cmp_df = pd.DataFrame({k: bt_stats(v) for k, v in runs.items()}).T
            cmp_df.index.name = "方法"
            st.dataframe(
                cmp_df.reset_index(), hide_index=True, width="stretch",
                column_config={
                    "交易次數": st.column_config.NumberColumn(format="%d"),
                    "勝率": st.column_config.NumberColumn(format="percent"),
                    "總損益": st.column_config.NumberColumn(format="%+.0f"),
                    "獲利因子": st.column_config.NumberColumn(format="%.2f"),
                    "平均每筆": st.column_config.NumberColumn(format="%+.1f"),
                    "最大單筆虧損": st.column_config.NumberColumn(format="%+.0f"),
                    "最大回撤": st.column_config.NumberColumn(format="%+.0f"),
                })
            st.caption(f"期間 {d5.index[0]:%Y-%m-%d} ～ {d5.index[-1]:%Y-%m-%d}，"
                       f"{len(set(d5.index.date))} 個交易日；樣本短，僅供參考。")

    if not bt.trades:
        st.info("這段期間沒有交易訊號。")
    else:
        tbl = bt.table()
        pnl = tbl["損益"].to_numpy()
        wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
        pf = wins.sum() / -losses.sum() if losses.sum() < 0 else np.inf

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("交易次數", len(pnl))
        k2.metric("勝率", f"{len(wins) / len(pnl):.1%}")
        k3.metric("總損益", f"{pnl.sum():+,.1f} 點")
        k4.metric("獲利因子", "∞" if np.isinf(pf) else f"{pf:.2f}")
        k5.metric("平均賺 / 賠", f"{wins.mean() if len(wins) else 0:+,.0f} / "
                                f"{losses.mean() if len(losses) else 0:+,.0f}")
        k6.metric("最大單筆虧損", f"{pnl.min():+,.1f}")

        eq = tbl[["出場時間", "損益"]].copy()
        eq["累積損益"] = eq["損益"].cumsum()
        eq["回撤"] = eq["累積損益"] - eq["累積損益"].cummax()
        fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                             vertical_spacing=0.05, subplot_titles=("累積損益（點）", "回撤"))
        fig3.add_scatter(x=eq["出場時間"], y=eq["累積損益"], mode="lines", name="累積損益",
                         line=dict(color=K_COLOR, width=2), row=1, col=1)
        fig3.add_scatter(x=eq["出場時間"], y=eq["回撤"], fill="tozeroy", mode="lines", name="回撤",
                         line=dict(color=DOWN), row=2, col=1)
        fig3.update_layout(height=420, showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig3)

        cA, cB = st.columns(2)
        with cA:
            st.markdown("**出場原因**")
            st.dataframe(tbl.groupby("出場原因")["損益"].agg(次數="count", 總損益="sum", 平均="mean")
                         .round(1).sort_values("次數", ascending=False), width="stretch")
        with cB:
            st.markdown("**多空分開看**")
            st.dataframe(tbl.groupby("方向")["損益"].agg(
                次數="count", 總損益="sum", 勝率=lambda s: f"{(s > 0).mean():.0%}").round(1),
                width="stretch")

        st.markdown("**交易明細**")
        st.dataframe(tbl.iloc[::-1], hide_index=True, width="stretch",
                     column_config={"損益": st.column_config.NumberColumn(format="%+.2f")})
        st.download_button("下載交易明細 CSV", tbl.to_csv(index=False).encode("utf-8-sig"),
                           file_name="ma13_trades.csv", mime="text/csv")


# ───────────────────────── 閉環系統（第 2～4 層） ─────────────────────────
def closed_loop_inputs():
    """回傳（日盤 5 分K、夜盤 K、日 K、說明）；沒有 5 分K 時回傳 None"""
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    if sj_m1 is not None:
        day, night = cl.split_sessions(sj_m1)
        if day.empty:
            return None
        return sjd.to_interval(day, "5m"), night, cl.daily_from_intraday(day), "永豐 1 分K（日盤／夜盤自動分開）"
    if source == "上傳 CSV":
        if interval != "5m":
            return None
        day, night = cl.split_sessions(raw_df)
        return day, night, cl.daily_from_intraday(day), "上傳的 5 分K"
    if not has_intraday:
        return None
    try:
        daily = fetch_yf(symbol, "1y", "1d")
    except Exception:  # noqa: BLE001
        daily = cl.daily_from_intraday(intraday)
    return intraday, empty, daily, "Yahoo 5 分K（近 60 天）＋日 K 1 年；Yahoo 沒有夜盤"


@st.cache_data(show_spinner="計算三週期共振與每日記錄…")
def cached_closed_loop(d5_: pd.DataFrame, night_: pd.DataFrame, daily_: pd.DataFrame, long_only_: bool,
                       counter_: bool, stop_: float | None, half_pts_: float | None, half_pct_: float,
                       cost_: float):
    f5, f30, fd = cl.build_frame(d5_, daily_)
    tr = cl.run(f5, long_only=long_only_, counter=counter_, stop=stop_, half_min_pct=half_pct_,
                cost=cost_, half_min_pts=half_pts_)
    return f5, tr, cl.rule_stats(tr), cl.daily_journal(f5, night_, fd, f30, tr)


with tab_loop:
    st.caption("老哥說的「閉環」：有開始也有結束、不會斷掉、可以一直跑。"
               "第 1 層（單一週期進出場）在「📊 回測」；這裡是第 2～4 層。")
    inputs = closed_loop_inputs()
    if inputs is None:
        st.info("閉環系統需要 5 分K（Yahoo 近 60 天、永豐 API，或上傳 5m 的 CSV）。")
    else:
        cl5, cl_night, cl_daily, cl_src = inputs
        o1, o2, o3 = st.columns(3)
        counter = o1.toggle("逆勢當沖（日線、30 分都反向時也做 5 分K 訊號）", value=True)
        half_mode = o2.radio("「回吐一半」啟動門檻", ["依側欄點數", "依進場價 %"], horizontal=True,
                             help="最高獲利超過門檻才啟動回吐一半就砍，避免小波動就出場")
        half_pct = o3.number_input("門檻（進場價 %）", 0.0, 5.0, 0.1, 0.05,
                                   disabled=half_mode != "依進場價 %")
        half_pts = min_profit if half_mode == "依側欄點數" else None

        f5c, trades_cl, stats_cl, journal_cl = cached_closed_loop(
            cl5, cl_night, cl_daily, long_only, counter, stop, half_pts, half_pct, cost)
        st.caption(f"資料：{cl_src}　{cl5.index[0]:%Y-%m-%d} ～ {cl5.index[-1]:%Y-%m-%d}　"
                   f"日 K {len(cl_daily)} 天　夜盤 {len(cl_night):,} 根")

        # ── 目前三週期狀態
        st.subheader("目前三週期狀態")
        r = f5c.iloc[-1]
        dd, m30, m5 = int(r["日線"]), int(r["三十分"]), int(r["五分"])
        icon = {1: "多 🔴", -1: "空 🟢", 0: "— ⚪"}
        s1, s2, s3 = st.columns(3)
        s1.metric("日線（前一日收盤）", icon[dd])
        s2.metric("30 分K", icon[m30])
        s3.metric("5 分K", icon[m5], f"{r['Close'] - r['MA13']:+,.1f} 距 13MA", delta_color="inverse")
        if dd == m30 == m5 != 0:
            st.success(f"**三週期共振（{cl.DIR_TXT[dd]}）**：可抱波段單", icon="🔁")
        elif dd == m30 != 0:
            st.warning(f"日線、30 分同為**{cl.DIR_TXT[dd]}**，5 分K 逆向：波段單減碼或觀望；逆向只能做當沖", icon="⚠️")
        else:
            st.info("週期不一致：只做當沖，收盤前一定平倉", icon="⏱️")

        # ── 第 2 層
        st.subheader("第 2 層・時間閉環：每日時間記錄")
        ok = (journal_cl["時間銜接檢查"] == "完整").mean() if len(journal_cl) else 0
        j1, j2, j3, j4 = st.columns(4)
        j1.metric("記錄天數", f"{len(journal_cl)} 天")
        j2.metric("時間銜接完整", f"{ok:.0%}")
        cover_col = f"推估幅度(×{OPEN_ENERGY_MULT})"
        j3.metric(f"第一根 ×{OPEN_ENERGY_MULT} 涵蓋實際幅度", f"{journal_cl['推估是否涵蓋'].mean():.0%}")
        j4.metric("缺口全補比例", f"{(journal_cl.get('缺口回補') == '全補').mean():.0%}"
                  if "缺口回補" in journal_cl else "—")
        if cl_night.empty:
            msg = ("夜盤沒有資料，所以每天都會標「無夜盤資料」——這就是第 2 層缺的那一段。")
            if source == "永豐 Shioaji":
                msg += "要記錄夜盤：期貨代號（如 TXFR1）並關掉側欄「期貨只取日盤」。"
            elif source == "Yahoo Finance":
                msg += "要記錄夜盤請改用永豐 API 抓期貨（如 TXFR1），並關掉「期貨只取日盤」。"
            st.info(msg, icon="🌙")
        jcols = ["日期", "星期", "昨收", "夜盤高", "夜盤低", "夜盤收", "夜盤收→開盤", "開盤", "跳空點", "跳空%",
                 "缺口大小", "第一根5分K振幅", cover_col, "實際幅度", "缺口回補", "收盤",
                 "收盤時日線", "收盤時30分", "收盤時5分", "當日交易筆數", "當日損益點", "時間銜接檢查"]
        jview = journal_cl[[c for c in jcols if c in journal_cl]].iloc[::-1]
        st.dataframe(jview, hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="%.0f") for c in
            ["昨收", "夜盤高", "夜盤低", "夜盤收", "開盤", "收盤", "第一根5分K振幅", cover_col, "實際幅度"]
        } | {"跳空點": st.column_config.NumberColumn(format="%+.0f"),
             "夜盤收→開盤": st.column_config.NumberColumn(format="%+.0f"),
             "跳空%": st.column_config.NumberColumn(format="%+.2f%%"),
             "當日損益點": st.column_config.NumberColumn(format="%+.1f")})

        # ── 第 3 層
        st.subheader("第 3 層・週期閉環：三週期共振回測")
        st.caption("進場：5 分K 多頭／空頭成立。日線＋30 分＋5 分同向 → **波段單**（30 分K 反向穿越 13MA 或日線翻轉才出，可留倉）；"
                   "其他 → **當沖單**（5 分K 反向穿越 13MA 出場，收盤前一定平倉）。"
                   f"全部：獲利回吐超過 {HALF_RETRACE:.0%} 就砍"
                   + (f"、停損 {stop:g} 點" if stop else "") + "。以 5 分K 收盤價成交，"
                   + (f"每筆扣 {cost:g} 點。" if cost else "未計成本。"))
        if trades_cl.empty:
            st.info("這段期間沒有交易訊號。")
        else:
            p = trades_cl["損益點"]
            w, l = p[p > 0], p[p <= 0]
            pf = w.sum() / -l.sum() if l.sum() < 0 else np.inf
            eqc = p.cumsum()
            t1, t2, t3, t4, t5, t6 = st.columns(6)
            t1.metric("交易次數", len(p))
            t2.metric("勝率", f"{len(w) / len(p):.1%}")
            t3.metric("總損益（點）", f"{p.sum():+,.0f}")
            t4.metric("獲利因子", "∞" if np.isinf(pf) else f"{pf:.2f}")
            t5.metric("最大單筆虧損", f"{p.min():+,.0f}")
            t6.metric("最大回撤", f"{(eqc - eqc.cummax()).min():+,.0f}")

            fig_l = go.Figure()
            fig_l.add_scatter(x=trades_cl["出場時間"], y=eqc, mode="lines", name="全部",
                              line=dict(color=K_COLOR, width=2.5))
            for typ, color in (("波段", UP), ("當沖", "#868e96")):
                sub = trades_cl[trades_cl["類型"] == typ]
                if len(sub):
                    fig_l.add_scatter(x=sub["出場時間"], y=sub["損益點"].cumsum(), mode="lines",
                                      name=f"{typ}單", line=dict(color=color, width=1.5, dash="dot"))
            fig_l.update_layout(height=380, hovermode="x unified", margin=dict(l=10, r=10, t=70, b=10),
                                title=dict(text="累積損益（點）：全部 vs 波段單 vs 當沖單", y=0.97),
                                legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"))
            st.plotly_chart(fig_l)

        # ── 第 4 層
        st.subheader("第 4 層・驗證閉環：規則統計（從尾到頭）")
        if not stats_cl.empty:
            st.caption("每筆交易都記下是哪條規則進場、哪條規則出場——哪一條規則在賺、哪一條在賠，一看就知道。")
            g1, g2 = st.columns(2)
            for col, grp, title in ((g1, "類型／進場規則", "依進場規則"), (g2, "出場規則", "依出場規則")):
                sub = stats_cl[stats_cl["分組"] == grp]
                fig_s = go.Figure(go.Bar(
                    x=sub["總損益點"], y=sub["項目"], orientation="h",
                    marker_color=np.where(sub["總損益點"] >= 0, UP, DOWN),
                    text=[f"{v:+,.0f}（{n} 筆・勝率 {wr:.0f}%）" for v, n, wr in
                          zip(sub["總損益點"], sub["筆數"], sub["勝率%"])],
                    textposition="auto"))
                fig_s.update_layout(height=60 + 46 * len(sub), title=f"{title}・總損益（點）",
                                    margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(autorange="reversed"))
                col.plotly_chart(fig_s)
            st.dataframe(stats_cl, hide_index=True, width="stretch", column_config={
                "勝率%": st.column_config.NumberColumn(format="%.1f%%"),
                "總損益點": st.column_config.NumberColumn(format="%+.1f"),
                "平均損益點": st.column_config.NumberColumn(format="%+.1f")})

            with st.expander("交易明細（每筆的進出場規則、三週期狀態、最大有利／不利）"):
                st.dataframe(trades_cl.iloc[::-1], hide_index=True, width="stretch",
                             column_config={"損益點": st.column_config.NumberColumn(format="%+.2f")})

        tag = cl.safe_name(symbol)
        b1, b2, b3 = st.columns(3)
        b1.download_button("下載每日時間記錄 CSV", journal_cl.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"每日時間記錄_{tag}.csv", mime="text/csv")
        b2.download_button("下載交易明細 CSV", trades_cl.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"交易明細_{tag}.csv", mime="text/csv", disabled=trades_cl.empty)
        b3.download_button("下載規則統計 CSV", stats_cl.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"規則統計_{tag}.csv", mime="text/csv", disabled=stats_cl.empty)
        st.caption("樣本短、未計實際成本與滑價，僅供檢討規則用，不構成投資建議。")


# ───────────────────────── 零股隔日沖 ─────────────────────────
@st.cache_data(show_spinner="計算隔日沖明細…")
def cached_overnight_table(data: pd.DataFrame, bar_minutes: int, hi_minutes: int) -> pd.DataFrame:
    return overnight_table(data, bar_minutes, hi_minutes)


with tab_odd:
    st.caption("今天收盤（或盤後零股）買進，隔天早上賣出。零股不能放空、盤中零股不能當沖，所以只做多、只留一夜。")
    _sym = symbol.strip().upper()
    if _sym.startswith(("^", "TSE", "OTC")) or (source == "永豐 Shioaji" and not _sym[:4].isdigit()):
        st.warning(f"{symbol} 是指數或期貨，無法買零股。以下僅供參考，請在左側改輸入個股或 ETF，"
                   "如 2454、2330、0050（在左側下拉選單選取即可）。", icon="⚠️")

    o1, o2, o3 = st.columns(3)
    if sj_m1 is not None:
        data_opts = {"5m": f"5 分K（永豐 {period}，出場可算到 9:10）", "60m": f"60 分K（永豐 {period}）"}
    else:
        data_opts = {"60m": "60 分K（約 3 年，樣本多）", "5m": "5 分K（60 天，出場可算到 9:10）"}
    odd_data = o1.radio("資料", list(data_opts), format_func=data_opts.get, key="odd_data")
    odd_filter = o2.selectbox("買進濾網（收盤時判斷）", list(OVERNIGHT_FILTERS), index=1,
                              format_func=OVERNIGHT_FILTERS.get, key="odd_filter",
                              help="盤中K：5 分K 資料時用 30 分K，60 分K 資料時用 60 分K")
    odd_exit = o3.selectbox("隔日出場", list(OVERNIGHT_EXITS), format_func=OVERNIGHT_EXITS.get,
                            key="odd_exit")
    gap_pct = 1.0
    if odd_exit == "gap":
        gap_pct = o3.number_input("大跳空門檻（%）", min_value=0.1, value=1.0, step=0.1, key="odd_gap")

    c_amt, c_disc, c_min, c_out = st.columns(4)
    amount = c_amt.number_input("每筆金額（元）", min_value=100, value=10_000, step=1_000, key="odd_amt")
    disc = c_disc.number_input("手續費折數", min_value=0.1, max_value=1.0, value=0.6, step=0.05,
                               key="odd_disc", help="0.6 = 6 折；未打折填 1")
    min_fee = c_min.number_input("最低手續費（元／邊）", min_value=0.0, value=1.0, step=1.0, key="odd_min",
                                 help="各券商零股最低手續費不同，常見 1 元或 20 元")
    cost_pct = odd_lot_cost_pct(amount, disc, min_fee)
    c_out.metric("來回成本", f"{cost_pct:.3f}%", help="手續費 0.1425%×折數×2 邊＋證交稅 0.3%（隔日沖不減半）")

    # 取資料
    odd_src = None
    if source == "上傳 CSV":
        if interval == odd_data:
            odd_src = raw_df
        else:
            st.info(f"上傳的 CSV 是 {interval}，請把「資料」切到相同週期（只支援 5m 或 60m）。")
    elif sj_m1 is not None:
        odd_src = sjd.to_interval(sj_m1, odd_data)
    else:
        try:
            odd_src = fetch_yf(symbol, "730d" if odd_data == "60m" else "60d", odd_data)
        except Exception as e:  # noqa: BLE001
            st.error(f"讀取 {odd_data} 資料失敗：{e}")

    if odd_src is not None and len(odd_src):
        bar_m = 60 if odd_data == "60m" else 5
        hi_m = 60 if odd_data == "60m" else 30
        otbl = cached_overnight_table(odd_src, bar_m, hi_m)
        if odd_data == "60m" and odd_exit in ("odd", "gap"):
            st.caption("60 分K 沒有 9:10 的價格，「零股第一盤」以隔日第一根 60 分K 收盤（10:00）代替；"
                       "要精確到 9:10 請切到 5 分K。")

        # 今晚訊號
        today = otbl.iloc[-1]
        ok = bool(today[odd_filter])
        flags = "　".join(f"{'✅' if today[k] else '❌'} {v}" for k, v in OVERNIGHT_FILTERS.items() if k != "none")
        (st.success if ok else st.info)(
            f"**{today['日期']} 收盤 {today['買進價']:,.2f}：**"
            + ("濾網成立，可在收盤／盤後零股買進，隔日出場" if ok else "濾網不成立，今晚不買")
            + f"\n\n{flags}", icon="🌙")

        res = overnight_returns(otbl, odd_filter, odd_exit, gap_pct, cost_pct)
        s = overnight_stats(res)
        if not s["交易次數"]:
            st.info("這個濾網在這段期間沒有任何買進日。")
        else:
            m = st.columns(6)
            m[0].metric("交易次數", f"{s['交易次數']}",
                        help=f"共 {otbl['r_open'].notna().sum()} 個交易日")
            m[1].metric("勝率（扣成本）", f"{s['勝率']:.0%}")
            m[2].metric("平均毛報酬", f"{s['平均毛報酬%']:+.2f}%")
            m[3].metric("平均淨報酬", f"{s['平均淨報酬%']:+.2f}%")
            m[4].metric("累積報酬（複利）", f"{s['累積報酬%']:+.1f}%")
            m[5].metric("最大回撤", f"{s['最大回撤%']:.1f}%")

            base = overnight_returns(otbl, "none", odd_exit, gap_pct, cost_pct)
            eq = go.Figure()
            for lbl, t, color in ((OVERNIGHT_FILTERS[odd_filter], res, K_COLOR),
                                  ("每天都買（對照）", base, "#adb5bd")):
                eq.add_scatter(x=pd.to_datetime(t["日期"]),
                               y=((1 + t["淨報酬%"] / 100).cumprod() - 1) * 100,
                               mode="lines", name=lbl, line=dict(color=color, width=2))
            eq.add_hline(y=0, line=dict(color="#868e96", dash="dot", width=1))
            eq.update_layout(height=360, title="累積淨報酬（%，複利）", hovermode="x unified",
                             margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.12))
            st.plotly_chart(eq)

            cmpA, cmpB = st.columns([3, 2])
            with cmpA:
                st.markdown(f"**濾網比較**（出場：{OVERNIGHT_EXITS[odd_exit]}，同樣成本）")
                cmp_rows = {v: overnight_stats(overnight_returns(otbl, k, odd_exit, gap_pct, cost_pct))
                            for k, v in OVERNIGHT_FILTERS.items()}
                cmp = pd.DataFrame(cmp_rows).T.reset_index(names="濾網")
                st.dataframe(cmp, hide_index=True, width="stretch", column_config={
                    "交易次數": st.column_config.NumberColumn(format="%d"),
                    "勝率": st.column_config.NumberColumn(format="percent"),
                    **{c: st.column_config.NumberColumn(format="%+.2f") for c in
                       ["平均毛報酬%", "平均淨報酬%", "中位數淨報酬%", "最差一筆%"]},
                    "累積報酬%": st.column_config.NumberColumn(format="%+.1f"),
                    "最大回撤%": st.column_config.NumberColumn(format="%.1f"),
                    "獲利因子": st.column_config.NumberColumn(format="%.2f"),
                })
            with cmpB:
                st.markdown("**依隔日跳空大小分組**（缺口要量化記錄）")
                bins = [-np.inf, -2, -1, 0, 1, 2, np.inf]
                labels = ["< −2%", "−2～−1%", "−1～0%", "0～1%", "1～2%", "> 2%"]
                g = res.groupby(pd.cut(res["跳空%"], bins, labels=labels), observed=False)["淨報酬%"]
                st.dataframe(pd.DataFrame({"次數": g.count(), "平均淨報酬%": g.mean().round(2)}),
                             width="stretch")

            with st.expander("交易明細"):
                show = res[["日期", "出場日", "買進價", "隔日開", "隔日零股第一盤", "隔日收",
                            "跳空%", "出場方式", "毛報酬%", "淨報酬%"]].iloc[::-1]
                st.dataframe(show, hide_index=True, width="stretch", column_config={
                    c: st.column_config.NumberColumn(format="%+.2f") for c in ["跳空%", "毛報酬%", "淨報酬%"]})
                st.download_button("下載隔日沖明細 CSV", show.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="odd_lot_overnight.csv", mime="text/csv")
            st.caption("買進價取當日最後一根 K 的收盤；Yahoo 為整股價格，零股成交價可能略有差異。"
                       "歷史回測不代表未來，不構成投資建議。")


# ───────────────────────── 費氏座標 ─────────────────────────
with tab_fib:
    st.caption(f"以最近 {fib_lookback} 根 K 棒：低 {fib_lo:,.2f} ～ 高 {fib_hi:,.2f}")
    f1, f2 = st.columns(2)
    with f1:
        st.markdown("**回檔與延伸關卡**")
        fib_df = pd.DataFrame({"關卡": list(fibs), "價位": list(fibs.values())})
        fib_df["距收盤"] = fib_df["價位"] - last["Close"]
        st.dataframe(fib_df, hide_index=True, width="stretch",
                     column_config={"價位": st.column_config.NumberColumn(format="%.2f"),
                                    "距收盤": st.column_config.NumberColumn(format="%+.2f")})
    with f2:
        st.markdown("**費氏數列座標（往上的關卡）**")
        seq = fib_sequence_levels(last["Close"], count=6)
        st.dataframe(pd.DataFrame({"關卡": seq, "距收盤": [s - last["Close"] for s in seq]}),
                     hide_index=True, width="stretch",
                     column_config={"關卡": st.column_config.NumberColumn(format="localized"),
                                    "距收盤": st.column_config.NumberColumn(format="%+.2f")})
        st.caption("例：80 → 130 → 210 → 340，數列乘上 10 的冪次。")


# ───────────────────────── 規則 ─────────────────────────
with tab_rules:
    st.markdown(f"""
1. **只信開盤與收盤**：開盤跳空＝非線性能量；能量＝缺口＋第一根 5 分 K 大小。
   第一根 5 分 K 振幅 × {OPEN_ENERGY_MULT} ≈ 當天可能的波動幅度。
2. **{MA_LEN}MA 是多空開關**：站上 13MA 且 KD>50、MACD 過 0 → 做多；跌破 13MA → 做空。
3. **K 線三根法則**：一根過前兩根高 → 開始動；一根破前兩根低 → 開始跌。
4. **回檔超過一半就砍**：獲利從最高回吐超過 {HALF_RETRACE:.0%} 出場。
5. **費氏座標**：0.382 / 0.5 / 0.618 回檔，費氏數列關卡（80→130→210…）。
6. **飆股**：KD>80 且低點不破 → 目標價 ×3（極端條件，僅作標記）。
7. **缺口要量化記錄**：跳多少點、補了一半還是全補。
8. **閉環**：有開始也有結束、不會斷掉、可以一直跑（見「🔁 閉環系統」）。
   時間要接起來（夜盤→開盤→收盤）、週期要共振（日線＋30 分＋5 分），每筆交易都要能從結果回推是哪條規則。

> 規則來源：老哥錄音談話整理。教學用規則化程式，未計手續費、滑價與稅，不構成投資建議。
""")
