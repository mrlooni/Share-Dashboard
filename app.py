import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="Share Dashboard", page_icon="📈", layout="wide")

# ───────────────────────── Theme ─────────────────────────
UP, DOWN, AMBER, INK, PANEL, MUTED = "#35D0A0", "#FF6B6B", "#FFB020", "#0F1420", "#171E2E", "#8A94A8"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Manrope', sans-serif; }}
.block-container {{ padding-top: 2rem; max-width: 1200px; }}
h1 {{ font-weight: 800; letter-spacing: -0.02em; }}
h2, h3 {{ font-weight: 700; letter-spacing: -0.01em; }}
[data-testid="stMetric"] {{ background: {PANEL}; border: 1px solid #232C42; border-radius: 12px; padding: 14px 16px; }}
[data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; font-weight: 700; }}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid #232C42; }}
.stTabs [data-baseweb="tab"] {{ padding: 8px 14px; border-radius: 8px 8px 0 0; }}
.stTabs [aria-selected="true"] {{ color: {AMBER}; }}
.banner {{ border-radius: 12px; padding: 14px 18px; font-weight: 700; margin: 8px 0 18px 0; }}
.banner small {{ font-weight: 500; opacity: .85; display:block; margin-top:2px; }}
.bull {{ background: rgba(53,208,160,.14); border: 1px solid {UP}; color: {UP}; }}
.bear {{ background: rgba(255,107,107,.14); border: 1px solid {DOWN}; color: {DOWN}; }}
.neut {{ background: rgba(255,176,32,.12); border: 1px solid {AMBER}; color: {AMBER}; }}
.rangebar {{ position: relative; height: 8px; border-radius: 4px; margin: 14px 0 6px 0;
  background: linear-gradient(90deg, {DOWN}, {AMBER}, {UP}); }}
.rangebar span {{ position: absolute; top: -5px; width: 4px; height: 18px; background: #fff; border-radius: 2px; }}
.check {{ padding: 6px 0; border-bottom: 1px solid #232C42; font-size: 0.95rem; }}
.ok {{ color: {UP}; font-weight: 700; }} .no {{ color: {DOWN}; font-weight: 700; }}
.news a {{ color: #E6EAF2; text-decoration: none; font-weight: 600; }}
.news a:hover {{ color: {AMBER}; }}
.news small {{ color: {MUTED}; }}
.muted {{ color: {MUTED}; font-size: 0.85rem; }}
</style>
""",
    unsafe_allow_html=True,
)

PERIOD_DAYS = {"1mo": 31, "3mo": 92, "6mo": 183, "1y": 366, "2y": 731, "5y": 1827}


# ───────────────────────── Helpers ─────────────────────────
def fmt_num(v, d=2):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.{d}f}"


def fmt_pct(v, d=1, scale=100):
    return "–" if v is None else f"{v * scale:.{d}f}%"


def fmt_big(v, cur="INR"):
    if v is None:
        return "–"
    if cur == "INR":
        cr = v / 1e7
        if abs(cr) >= 1e5:
            return f"₹{cr / 1e5:,.2f} Lakh Cr"
        return f"₹{cr:,.0f} Cr"
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if abs(v) >= div:
            return f"{v / div:,.2f}{unit} {cur}"
    return f"{v:,.0f} {cur}"


@st.cache_data(ttl=600, show_spinner=False)
def search_symbols(q):
    q = q.strip()
    if not q:
        return []
    if re.fullmatch(r"[\^A-Za-z0-9\-&]+\.[A-Za-z]{1,3}|\^[A-Za-z0-9]+", q):
        return [(q.upper(), q.upper(), "")]
    out = []
    try:
        for r in yf.Search(q, max_results=10).quotes:
            if r.get("quoteType") in ("EQUITY", "ETF", "INDEX"):
                out.append((r["symbol"], r.get("longname") or r.get("shortname") or r["symbol"], r.get("exchDisp", "")))
    except Exception:
        pass
    out.sort(key=lambda x: 0 if x[0].endswith(".NS") else 1 if x[0].endswith(".BO") else 2)
    return out or [(q.upper() + ".NS", q.upper(), "NSE")]


@st.cache_data(ttl=300, show_spinner=False)
def load_history(sym):
    df = yf.Ticker(sym).history(period="5y", interval="1d")
    if df.empty:
        return df
    df.index = df.index.tz_localize(None)
    c = df["Close"]
    df["SMA20"], df["SMA50"], df["SMA200"] = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    d = c.diff()
    rs = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean() / (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + rs)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df


@st.cache_data(ttl=900, show_spinner=False)
def load_info(sym):
    try:
        return yf.Ticker(sym).info or {}
    except Exception:
        return {}


@st.cache_data(ttl=900, show_spinner=False)
def load_statements(sym):
    t = yf.Ticker(sym)
    out = {}
    for k, attr in (("income", "financials"), ("q_income", "quarterly_financials"),
                    ("balance", "balance_sheet"), ("cashflow", "cashflow")):
        try:
            out[k] = getattr(t, attr)
        except Exception:
            out[k] = pd.DataFrame()
    return out


@st.cache_data(ttl=900, show_spinner=False)
def load_news(sym):
    try:
        return yf.Ticker(sym).news or []
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def quick_quote(sym):
    try:
        h = yf.Ticker(sym).history(period="5d")["Close"].dropna()
        return float(h.iloc[-1]), float(h.iloc[-1] / h.iloc[-2] - 1)
    except Exception:
        return None, None


def slice_period(df, period):
    return df.loc[df.index >= df.index[-1] - pd.Timedelta(days=PERIOD_DAYS[period])]


def plotly_layout(fig, h=420):
    fig.update_layout(
        height=h, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope", color="#E6EAF2"), hovermode="x unified",
        legend=dict(orientation="h", y=1.05, x=0),
    )
    fig.update_xaxes(gridcolor="#1E2638", showline=False)
    fig.update_yaxes(gridcolor="#1E2638", showline=False)
    return fig


# ───────────────────────── Sidebar: watchlist ─────────────────────────
if "watch" not in st.session_state:
    st.session_state.watch = ["RELIANCE.NS", "TCS.NS", "^NSEI"]

with st.sidebar:
    st.markdown("### Watchlist")
    for w in list(st.session_state.watch):
        p, ch = quick_quote(w)
        c1, c2, c3 = st.columns([3, 3, 1])
        c1.markdown(f"**{w.replace('.NS', '')}**")
        if p is not None:
            col = UP if ch >= 0 else DOWN
            c2.markdown(f"<span style='color:{col}'>{p:,.1f}<br><small>{ch * 100:+.2f}%</small></span>", unsafe_allow_html=True)
        if c3.button("✕", key=f"rm_{w}"):
            st.session_state.watch.remove(w)
            st.rerun()
    st.caption("Add the stock you are viewing from the Overview tab.")

# ───────────────────────── Header + inputs ─────────────────────────
st.title("📈 Share Dashboard")

c1, c2 = st.columns([3, 2])
query = c1.text_input("Share name / symbol", value="Reliance", help="Try: Reliance, TCS, HDFCBANK.NS, AAPL")
period = c2.radio("Period", list(PERIOD_DAYS), index=3, horizontal=True)

matches = search_symbols(query)
labels = [f"{n.upper()} · {s}" + (f" · {e}" if e else "") for s, n, e in matches]
choice = st.selectbox("Select match", labels)
symbol = matches[labels.index(choice)][0]

df_all = load_history(symbol)
if df_all.empty:
    st.error("No price data found for this symbol. Check the name or try the exact ticker, e.g. TCS.NS")
    st.stop()

info = load_info(symbol)
cur = info.get("currency", "INR")
df = slice_period(df_all, period)
last, prev = df_all["Close"].iloc[-1], df_all["Close"].iloc[-2]
chg, chg_pct = last - prev, (last / prev - 1) * 100

st.subheader(info.get("longName") or choice.split(" · ")[0].title())
m1, m2, m3, m4 = st.columns(4)
m1.metric("Price", f"{last:,.2f} {cur}", f"{chg:+.2f} ({chg_pct:+.2f}%)")
m2.metric("Volume", f"{int(df_all['Volume'].iloc[-1]):,}")
m3.metric("Market cap", fmt_big(info.get("marketCap"), cur))
m4.metric("P/E (trailing)", fmt_num(info.get("trailingPE")))

# ───────────────────────── Trend signal ─────────────────────────
row = df_all.iloc[-1]
rsi = row["RSI"]
above50, above200 = last > row["SMA50"], (pd.isna(row["SMA200"]) or last > row["SMA200"])
if above50 and above200 and row["MACD"] > row["Signal"]:
    cls, head = "bull", "Bullish trend: look for entries"
elif not above50 and not above200:
    cls, head = "bear", "Bearish trend: wait / avoid"
else:
    cls, head = "neut", "Mixed trend: wait for confirmation"
note = f"RSI {rsi:.0f}" + (" (overbought)" if rsi > 70 else " (oversold)" if rsi < 30 else "")
note += f" · {'above' if above50 else 'below'} 50-day average · {'above' if above200 else 'below'} 200-day average"
st.markdown(f"<div class='banner {cls}'>{head}<small>{note}</small></div>", unsafe_allow_html=True)

tabs = st.tabs(["Overview", "Chart", "Fundamentals", "Financials", "Returns", "Compare", "News"])

# ───────────────────────── Overview ─────────────────────────
with tabs[0]:
    support = df_all["Low"].tail(20).min()
    resistance = df_all["High"].tail(60).max()
    entry, sl = last, support
    risk = max(entry - sl, entry * 0.005)
    t1, t2 = entry + 1.5 * risk, entry + 3 * risk

    top = st.columns([4, 1])
    top[0].markdown("### 🎯 Entry / Exit")
    if top[1].button("★ Add to watchlist") and symbol not in st.session_state.watch:
        st.session_state.watch.append(symbol)
        st.rerun()

    fig = go.Figure()
    fig.add_shape(type="rect", x0=sl, x1=entry, y0=0, y1=1, fillcolor="rgba(255,107,107,.25)", line_width=0)
    fig.add_shape(type="rect", x0=entry, x1=t2, y0=0, y1=1, fillcolor="rgba(53,208,160,.22)", line_width=0)
    pts = [("SL", sl, DOWN), ("Entry", entry, AMBER), ("T1", t1, UP), ("T2", t2, UP)]
    fig.add_trace(go.Scatter(
        x=[p[1] for p in pts], y=[0.5] * 4, mode="markers+text",
        marker=dict(size=14, color=[p[2] for p in pts]),
        text=[f"{p[0]}<br>{p[1]:,.1f}" for p in pts], textposition="top center", hoverinfo="skip",
    ))
    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_layout(showlegend=False)
    st.plotly_chart(plotly_layout(fig, 170), use_container_width=True)

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Entry", f"{entry:,.2f}")
    e2.metric("Stop-loss", f"{sl:,.2f}", f"-{(entry - sl) / entry * 100:.2f}%", delta_color="inverse")
    e3.metric("Target 1 (1:1.5)", f"{t1:,.2f}", f"+{(t1 / entry - 1) * 100:.2f}%")
    e4.metric("Target 2 (1:3)", f"{t2:,.2f}", f"+{(t2 / entry - 1) * 100:.2f}%")
    st.markdown(f"<span class='muted'>Support {support:,.1f} · Resistance {resistance:,.1f}</span>", unsafe_allow_html=True)

    hi = info.get("fiftyTwoWeekHigh") or df_all["High"].tail(252).max()
    lo = info.get("fiftyTwoWeekLow") or df_all["Low"].tail(252).min()
    pos = min(max((last - lo) / (hi - lo), 0), 1) * 100 if hi > lo else 50
    st.markdown("**52-week range**")
    st.markdown(f"<div class='rangebar'><span style='left:calc({pos:.1f}% - 2px)'></span></div>", unsafe_allow_html=True)
    st.markdown(f"<span class='muted'>{lo:,.1f} (low) ← {last:,.1f} ({pos:.0f}% of range) → {hi:,.1f} (high)</span>", unsafe_allow_html=True)

    st.markdown("### 🏢 Company")
    company = pd.DataFrame({
        "Field": ["Sector", "Industry", "Market cap", "P/E (trailing)", "EPS", "Dividend yield", "Beta", "Avg volume"],
        "Value": [
            info.get("sector", "–"), info.get("industry", "–"), fmt_big(info.get("marketCap"), cur),
            fmt_num(info.get("trailingPE")), fmt_num(info.get("trailingEps")),
            "–" if info.get("dividendYield") is None else f"{info['dividendYield']:.2f}%",
            fmt_num(info.get("beta"), 2), fmt_num(info.get("averageVolume"), 0),
        ],
    })
    st.dataframe(company, hide_index=True, use_container_width=True)
    with st.expander("About the company"):
        st.write(info.get("longBusinessSummary", "No description available."))
        if info.get("website"):
            st.markdown(f"[{info['website']}]({info['website']})")

# ───────────────────────── Chart ─────────────────────────
with tabs[1]:
    o1, o2, o3 = st.columns(3)
    kind = o1.radio("Style", ["Candles", "Line"], horizontal=True)
    smas = o2.multiselect("Averages", ["SMA20", "SMA50", "SMA200"], default=["SMA50", "SMA200"])
    panels = o3.multiselect("Panels", ["Volume", "RSI", "MACD"], default=["Volume", "RSI"])

    rows = 1 + len(panels)
    heights = [0.55] + [0.45 / max(len(panels), 1)] * len(panels)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=heights)
    if kind == "Candles":
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            increasing_line_color=UP, decreasing_line_color=DOWN, name="Price"), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], line=dict(color=AMBER, width=2), name="Close"), row=1, col=1)
    for s, colr in zip(("SMA20", "SMA50", "SMA200"), ("#7AA2F7", "#C792EA", "#FFFFFF")):
        if s in smas:
            fig.add_trace(go.Scatter(x=df.index, y=df[s], line=dict(width=1.3, color=colr), name=s), row=1, col=1)
    r = 2
    for p in panels:
        if p == "Volume":
            colors = np.where(df["Close"] >= df["Open"], UP, DOWN)
            fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=colors, name="Volume"), row=r, col=1)
        elif p == "RSI":
            fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color=AMBER), name="RSI"), row=r, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color=DOWN, row=r, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color=UP, row=r, col=1)
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], line=dict(color="#7AA2F7"), name="MACD"), row=r, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["Signal"], line=dict(color=AMBER), name="Signal"), row=r, col=1)
        r += 1
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(plotly_layout(fig, 520 + 90 * len(panels)), use_container_width=True)
    st.download_button("⬇ Download price data (CSV)", df.round(2).to_csv().encode(), f"{symbol}_{period}.csv", "text/csv")

# ───────────────────────── Fundamentals ─────────────────────────
with tabs[2]:
    g = info.get

    checks = [
        ("P/E below 25", g("trailingPE"), lambda v: v < 25),
        ("P/B below 4", g("priceToBook"), lambda v: v < 4),
        ("ROE above 15%", g("returnOnEquity"), lambda v: v > 0.15),
        ("Profit margin above 10%", g("profitMargins"), lambda v: v > 0.10),
        ("Debt/Equity below 100", g("debtToEquity"), lambda v: v < 100),
        ("Current ratio above 1", g("currentRatio"), lambda v: v > 1),
        ("Revenue growing", g("revenueGrowth"), lambda v: v > 0),
        ("Earnings growing", g("earningsGrowth"), lambda v: v > 0),
        ("Pays a dividend", g("dividendYield"), lambda v: v > 0),
    ]
    scored = [(n, v, f(v)) for n, v, f in checks if v is not None]
    if scored:
        score = round(sum(1 for *_, ok in scored if ok) / len(scored) * 100)
        s1, s2 = st.columns([1, 2])
        s1.metric("Fundamental score", f"{score} / 100")
        s1.progress(score / 100)
        s1.caption("Simple rule-based checklist. Compare within the same sector.")
        with s2:
            for n, v, ok in scored:
                mark = "<span class='ok'>✓</span>" if ok else "<span class='no'>✗</span>"
                st.markdown(f"<div class='check'>{mark}&nbsp; {n} <span class='muted'>({fmt_num(v)})</span></div>", unsafe_allow_html=True)
    else:
        st.info("Fundamental data is not available for this symbol.")

    def block(title, items):
        st.markdown(f"#### {title}")
        cols = st.columns(len(items))
        for col, (lbl, val) in zip(cols, items):
            col.metric(lbl, val)

    block("Valuation", [("P/E (trailing)", fmt_num(g("trailingPE"))), ("P/E (forward)", fmt_num(g("forwardPE"))),
                        ("P/B", fmt_num(g("priceToBook"))), ("P/S", fmt_num(g("priceToSalesTrailing12Months"))),
                        ("EV/EBITDA", fmt_num(g("enterpriseToEbitda")))])
    block("Profitability", [("Gross margin", fmt_pct(g("grossMargins"))), ("Operating margin", fmt_pct(g("operatingMargins"))),
                            ("Net margin", fmt_pct(g("profitMargins"))), ("ROE", fmt_pct(g("returnOnEquity"))),
                            ("ROA", fmt_pct(g("returnOnAssets")))])
    block("Growth", [("Revenue growth (YoY)", fmt_pct(g("revenueGrowth"))), ("Earnings growth (YoY)", fmt_pct(g("earningsGrowth"))),
                     ("EPS (trailing)", fmt_num(g("trailingEps"))), ("EPS (forward)", fmt_num(g("forwardEps"))),
                     ("Revenue", fmt_big(g("totalRevenue"), cur))])
    block("Balance sheet", [("Total cash", fmt_big(g("totalCash"), cur)), ("Total debt", fmt_big(g("totalDebt"), cur)),
                            ("Debt / Equity", fmt_num(g("debtToEquity"))), ("Current ratio", fmt_num(g("currentRatio"))),
                            ("Free cash flow", fmt_big(g("freeCashflow"), cur))])
    block("Dividends and ownership", [
        ("Dividend yield", "–" if g("dividendYield") is None else f"{g('dividendYield'):.2f}%"),
        ("Payout ratio", fmt_pct(g("payoutRatio"))), ("Book value", fmt_num(g("bookValue"))),
        ("Promoter / insider", fmt_pct(g("heldPercentInsiders"))), ("Institutions", fmt_pct(g("heldPercentInstitutions")))])

    if g("targetMeanPrice"):
        st.markdown("#### Analyst view")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Recommendation", str(g("recommendationKey", "–")).replace("_", " ").title())
        a2.metric("Mean target", f"{g('targetMeanPrice'):,.1f}", f"{(g('targetMeanPrice') / last - 1) * 100:+.1f}% vs price")
        a3.metric("Low target", fmt_num(g("targetLowPrice"), 1))
        a4.metric("High target", fmt_num(g("targetHighPrice"), 1))
        st.caption(f"Based on {g('numberOfAnalystOpinions', '–')} analyst opinions.")

# ───────────────────────── Financials ─────────────────────────
with tabs[3]:
    st_data = load_statements(symbol)
    view = st.radio("View", ["Annual", "Quarterly"], horizontal=True)
    inc = st_data["income"] if view == "Annual" else st_data["q_income"]

    if inc is not None and not inc.empty:
        inc = inc.sort_index(axis=1)
        fig = go.Figure()
        for rowname, colr in (("Total Revenue", "#7AA2F7"), ("Operating Income", AMBER), ("Net Income", UP)):
            if rowname in inc.index:
                s = inc.loc[rowname].dropna() / 1e7
                fig.add_trace(go.Bar(x=[d.strftime("%b %Y") for d in s.index], y=s.values, name=rowname, marker_color=colr))
        fig.update_layout(barmode="group", yaxis_title="₹ Crore" if cur == "INR" else "Millions" )
        if cur != "INR":
            for t in fig.data:
                t.y = np.array(t.y) * 1e7 / 1e6
        st.plotly_chart(plotly_layout(fig, 360), use_container_width=True)

    def show_table(title, d):
        if d is None or d.empty:
            return
        with st.expander(title):
            div = 1e7 if cur == "INR" else 1e6
            t = (d.sort_index(axis=1) / div).round(1)
            t.columns = [c.strftime("%b %Y") for c in t.columns]
            st.dataframe(t, use_container_width=True)
            st.caption("₹ Crore" if cur == "INR" else "Millions")

    show_table("Income statement", st_data["income"] if view == "Annual" else st_data["q_income"])
    show_table("Balance sheet", st_data["balance"])
    show_table("Cash flow", st_data["cashflow"])
    if all(x is None or x.empty for x in st_data.values()):
        st.info("Financial statements are not available for this symbol.")

# ───────────────────────── Returns ─────────────────────────
with tabs[4]:
    close = df_all["Close"]

    def ret(days):
        s = close[close.index <= close.index[-1] - pd.Timedelta(days=days)]
        return None if s.empty else (close.iloc[-1] / s.iloc[-1] - 1) * 100

    labels_r = {"1 week": 7, "1 month": 30, "3 months": 91, "6 months": 182, "1 year": 365, "3 years": 1095}
    cols = st.columns(len(labels_r))
    for col, (lbl, d) in zip(cols, labels_r.items()):
        v = ret(d)
        col.metric(lbl, "–" if v is None else f"{v:+.1f}%")

    daily = close.pct_change().dropna()
    dd = close / close.cummax() - 1
    v1, v2, v3 = st.columns(3)
    v1.metric("Volatility (annual)", f"{daily.tail(252).std() * np.sqrt(252) * 100:.1f}%")
    v2.metric("Max drawdown (5y)", f"{dd.min() * 100:.1f}%")
    v3.metric("Best day (1y)", f"{daily.tail(252).max() * 100:+.1f}%")

    fig = go.Figure(go.Scatter(x=dd.index, y=dd * 100, fill="tozeroy", line=dict(color=DOWN), name="Drawdown %"))
    fig.update_layout(title="Drawdown from peak (%)", title_font_size=14)
    st.plotly_chart(plotly_layout(fig, 260), use_container_width=True)

    monthly = close.resample("ME").last().pct_change().dropna() * 100
    if not monthly.empty:
        heat = monthly.to_frame("r")
        heat["Year"], heat["Month"] = heat.index.year, heat.index.strftime("%b")
        pivot = heat.pivot(index="Year", columns="Month", values="r").reindex(
            columns=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index.astype(str), zmid=0,
            colorscale=[[0, DOWN], [0.5, PANEL], [1, UP]], text=np.round(pivot.values, 1),
            texttemplate="%{text}", hovertemplate="%{y} %{x}: %{z:.1f}%<extra></extra>", showscale=False))
        fig.update_layout(title="Monthly returns (%)", title_font_size=14)
        st.plotly_chart(plotly_layout(fig, 90 + 45 * len(pivot)), use_container_width=True)

# ───────────────────────── Compare ─────────────────────────
with tabs[5]:
    txt = st.text_input("Compare with (comma separated)", value="^NSEI, TCS.NS")
    syms = [symbol] + [s.strip() for s in txt.split(",") if s.strip()]
    syms = [s if ("." in s or s.startswith("^")) else s.upper() + ".NS" for s in syms]
    fig = go.Figure()
    for s in dict.fromkeys(syms):
        h = load_history(s)
        if h.empty:
            st.warning(f"No data for {s}")
            continue
        h = slice_period(h, period)["Close"]
        fig.add_trace(go.Scatter(x=h.index, y=h / h.iloc[0] * 100, name=s, line=dict(width=2.5 if s == symbol else 1.5)))
    fig.update_layout(yaxis_title="Rebased to 100")
    st.plotly_chart(plotly_layout(fig, 420), use_container_width=True)

# ───────────────────────── News ─────────────────────────
with tabs[6]:
    items = load_news(symbol)
    if not items:
        st.info("No recent news found.")
    for n in items[:12]:
        c = n.get("content", n)
        title = c.get("title")
        url = (c.get("canonicalUrl") or {}).get("url") or c.get("link")
        src = (c.get("provider") or {}).get("displayName") or c.get("publisher", "")
        when = (c.get("pubDate") or "")[:10]
        if title and url:
            st.markdown(f"<div class='news check'><a href='{url}' target='_blank'>{title}</a><br><small>{src} · {when}</small></div>", unsafe_allow_html=True)

st.caption("Technical estimates and ratios are for learning only, not investment advice. Data: Yahoo Finance via yfinance.")
