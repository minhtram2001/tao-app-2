import streamlit as st
import warnings
import json
import sys
import pickle
import io
import numpy as np
import pandas as pd
import nevergrad as ng
from scipy.stats import ttest_1samp, wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# Set Page Config
st.set_page_config(
    page_title="Hệ Thống Tối Ưu Hóa Danh Mục Đầu Tư Chứng Khoán HOSE",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 38px;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 5px;
        text-align: center;
    }
    
    .subtitle {
        font-size: 18px;
        color: #4B5563;
        margin-bottom: 25px;
        text-align: center;
    }
    
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border-left: 5px solid #2563EB;
        text-align: center;
    }
    
    .metric-card-best {
        background-color: #EEF2FF;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.15);
        border: 2px solid #3B82F6;
        border-left: 5px solid #F59E0B;
        text-align: center;
    }
    
    .metric-title {
        font-size: 14px;
        font-weight: 600;
        color: #6B7280;
        margin-bottom: 5px;
    }
    
    .metric-val {
        font-size: 22px;
        font-weight: 700;
        color: #1F2937;
    }
    
    .metric-val-best {
        font-size: 24px;
        font-weight: 700;
        color: #1E3A8A;
    }
    
    .tag-best {
        background-color: #F59E0B;
        color: white;
        font-size: 10px;
        font-weight: bold;
        padding: 2px 6px;
        border-radius: 4px;
        position: relative;
        top: -10px;
    }
</style>
""", unsafe_allow_html=True)

# Try importing INDUSTRY_TICKERS from industry_tickers.py
try:
    from industry_tickers import INDUSTRY_TICKERS
except ImportError:
    # Fallback to definition if not found
    INDUSTRY_TICKERS = {
        "Thép": ['BVG','TTS','DTL','CBI','VDT','VGS','TMG','VGL','HLA','BCA','TVN','HPG','HMC','HMG','HSG','ITQ','KMT','KVC','MEL','MHL','POM','NKG','KKC','SMC','SHA','SSM','TDS','PAS','SHI','VCA','TTH','TLH','TNA','TNB','TNS','TNI','GDA','KTL','HSV','DFC','TIS'],
        "Ngân hàng": ['ACB', 'BAB', 'BID', 'CTG', 'EIB', 'EVF', 'HDB', 'KLB', 'LPB', 'MBB', 'MSB', 'NAB', 'NVB', 'OCB', 'SHB', 'SSB', 'STB', 'TCB', 'TPB', 'VAB', 'VCB', 'VIB', 'VPB'],
        "Bất động sản": ['AGG', 'API', 'BCM', 'CEO', 'CRE', 'DIG', 'DRH', 'DXG', 'DXS', 'HDC', 'HDG', 'IJC', 'KBC', 'KDH', 'KHG', 'LDG', 'NHA', 'NLG', 'NVL', 'PDR', 'SCR', 'SJS', 'SZC', 'TCH', 'VHM', 'VIC', 'VRE'],
        "Chứng khoán": ['AGR', 'APG', 'BSI', 'BVS', 'CTS', 'DSC', 'FTS', 'HCM', 'IVS', 'MBS', 'ORS', 'SHS', 'SSI', 'TCI', 'TVB', 'TVS', 'VCI', 'VDS', 'VIX', 'VND'],
        "Dầu khí / Khai khoáng": ['GAS', 'PLX', 'PVD', 'PVS', 'PVC', 'PVB', 'BSR', 'OIL', 'PVT', 'CNG', 'KSB', 'PVC', 'PVD', 'PVS'],
        "Bán lẻ / Bán buôn": ['MWG', 'FRT', 'DGW', 'PNJ', 'PET', 'PSD', 'HAX', 'SVC'],
        "Công nghệ": ['FPT', 'CMG', 'ELC', 'ITD', 'ICT', 'VGI', 'CTR', 'FOX'],
        "Thực phẩm - Đồ uống": ['VNM', 'SAB', 'MSN', 'MCH', 'KDC', 'DBC', 'BAF', 'PAN', 'VHC', 'ANV', 'IDI', 'HAG', 'SBT', 'QNS']
    }

# ==========================================================================
# CONFIG CLASS FOR STREAMLIT STATE
# ==========================================================================
class Config:
    def __init__(self, initial_capital, fee_rate, stop_loss_pct, num_stocks, pso_budget, 
                 lookback_sess, vol_lookback, trend_window, random_seed, score_w_sharpe, 
                 score_w_return, invest_years, rebalance_months, weight_scheme):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.stop_loss_pct = stop_loss_pct
        self.num_stocks = num_stocks
        self.pso_budget = pso_budget
        self.lookback_sess = lookback_sess
        self.vol_lookback = vol_lookback
        self.trend_window = trend_window
        self.random_seed = random_seed
        self.score_w_sharpe = score_w_sharpe
        self.score_w_return = score_w_return
        self.invest_years = invest_years
        self.rebalance_months = rebalance_months
        self.weight_scheme = weight_scheme

# ==========================================================================
# 1. TAI & CHUAN HOA DU LIEU
# ==========================================================================
@st.cache_data
def load_data(file_or_path):
    df = pd.read_csv(file_or_path, low_memory=False)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Parse dates with different potential formats
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    if df["date"].isna().all():
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    if df["date"].isna().any():
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["date"])
    
    use_adj = all(c in df.columns for c in ["adj_open", "adj_close"])
    df["Open"]  = df["adj_open"]  if use_adj else df["open"]
    df["Close"] = df["adj_close"] if use_adj else df["close"]
    df = df.rename(columns={"date": "Date", "ticker": "Ticker"})
    df = df[["Date", "Ticker", "Open", "Close"]].dropna(subset=["Open", "Close"])
    df = df[df["Close"] > 0]
    return df.sort_values(["Ticker", "Date"]).reset_index(drop=True), use_adj

# ==========================================================================
# 2. CHI BAO KY THUAT
# ==========================================================================
def rsi_wilder(close, period):
    delta = close.diff(); gain = delta.clip(lower=0); loss = (-delta).clip(lower=0)
    ag = gain.ewm(com=period - 1, min_periods=period).mean()
    al = loss.ewm(com=period - 1, min_periods=period).mean()
    return 100 - (100 / (1 + ag / al.replace(0, np.nan)))

def macd_lines(close, fast, slow, sign):
    macd = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    return macd, macd.ewm(span=sign, adjust=False).mean()

def bollinger(close, window, dev):
    ma = close.rolling(window, min_periods=window).mean()
    sd = close.rolling(window, min_periods=window).std()
    return ma + dev * sd, ma - dev * sd

# ==========================================================================
# 3. THAM SO MAC DINH & MIEN PSO
# ==========================================================================
def default_params(strat):
    if strat == "rsi":           return dict(rsi_window=14, lower=30, upper=70)
    if strat == "rsi_macd":      return dict(rsi_window=14, upper=70, macd_fast=12, macd_slow=26, macd_sign=9)
    if strat == "bollinger_rsi": return dict(bb_window=20, bb_dev=2.0, rsi_window=14, lower=45, upper=70)
    raise ValueError(strat)

def parametrization(strat):
    I = lambda lo, hi: ng.p.Scalar(lower=lo, upper=hi).set_integer_casting()
    F = lambda lo, hi: ng.p.Scalar(lower=lo, upper=hi)
    if strat == "rsi":
        return ng.p.Dict(rsi_window=I(5, 30), lower=F(15, 40), upper=F(60, 85))
    if strat == "rsi_macd":
        return ng.p.Dict(rsi_window=I(5, 30), upper=F(55, 85), macd_fast=I(5, 20), macd_slow=I(21, 40), macd_sign=I(5, 15))
    if strat == "bollinger_rsi":
        return ng.p.Dict(bb_window=I(10, 30), bb_dev=F(1.5, 3.0), rsi_window=I(5, 25), lower=F(30, 55), upper=F(60, 85))
    raise ValueError(strat)

def clean_params(strat, p):
    p = dict(p)
    for k in ["rsi_window", "bb_window", "macd_fast", "macd_slow", "macd_sign"]:
        if k in p: p[k] = int(round(float(p[k])))
    for k in ["bb_dev", "lower", "upper"]:
        if k in p: p[k] = float(p[k])
    if "macd_fast" in p and "macd_slow" in p and p["macd_fast"] >= p["macd_slow"]:
        p["macd_slow"] = p["macd_fast"] + 5
    if "lower" in p and "upper" in p and p["lower"] >= p["upper"]:
        p["lower"], p["upper"] = min(p["lower"], p["upper"]), max(p["lower"], p["upper"]) + 5
    return p

# ==========================================================================
# 4. TIN HIEU MUA/BAN
# ==========================================================================
def generate_signals(df, strat, params, config):
    p = clean_params(strat, params); c = df["Close"]
    trend = c > c.rolling(config.trend_window, min_periods=config.trend_window).mean()
    if strat == "rsi":
        r = rsi_wilder(c, p["rsi_window"]); buy, sell = r < p["lower"], r > p["upper"]
    elif strat == "rsi_macd":
        r = rsi_wilder(c, p["rsi_window"]); m, s = macd_lines(c, p["macd_fast"], p["macd_slow"], p["macd_sign"])
        buy = (m > s) & (r < p["upper"]) & trend; sell = (m < s) | (r > p["upper"])
    elif strat == "bollinger_rsi":
        ub, lb = bollinger(c, p["bb_window"], p["bb_dev"]); r = rsi_wilder(c, p["rsi_window"])
        buy = (c < lb) & (r < p["lower"]) & trend; sell = (c > ub) | (r > p["upper"])
    else:
        raise ValueError(strat)
    return buy.fillna(False).values, sell.fillna(False).values

def signal_arrays(df, strat, params, config):
    buy, sell = generate_signals(df, strat, params, config)
    o = df["Open"].values.astype(float); c = df["Close"].values.astype(float)
    dt = df["Date"].values.astype("datetime64[ns]")
    buy_act = np.empty(len(buy), bool);  buy_act[0] = False;  buy_act[1:] = buy[:-1]
    sell_act = np.empty(len(sell), bool); sell_act[0] = False; sell_act[1:] = sell[:-1]
    cp = np.empty(len(c)); cp[0] = np.nan; cp[1:] = c[:-1]
    return dt, o, c, buy_act, sell_act, cp

# ==========================================================================
# 5. BACKTEST CORE
# ==========================================================================
def sim_core(arrays, start, end, capital, fee, sl, record=False):
    dt, o, c, buy_act, sell_act, cp = arrays
    idx = np.nonzero((dt >= np.datetime64(start)) & (dt < np.datetime64(end)))[0]
    if len(idx) < 2:
        return None, []
    cash, shares, in_pos, bp = float(capital), 0.0, False, 0.0
    vals = np.empty(len(idx)); trades = []
    for j in range(len(idx)):
        i = idx[j]; oi = o[i]; ci = c[i]; cpi = cp[i]
        if (not in_pos) and buy_act[i] and oi > 0 and cash > 0:
            shares = cash * (1 - fee) / oi; bp = oi; cash = 0.0; in_pos = True
            if record: trades.append({"Type": "MUA", "Date": dt[i], "Price": float(oi)})
        elif in_pos:
            stop = (not np.isnan(cpi)) and (cpi < bp * (1 - sl))
            if sell_act[i] or stop:
                proceeds = shares * oi * (1 - fee)
                if record: trades.append({"Type": "BAN", "Date": dt[i], "Price": float(oi),
                                          "PnL": float(proceeds - shares * bp),
                                          "Reason": "STOP-LOSS" if (stop and not sell_act[i]) else "TIN HIEU"})
                cash = proceeds; shares = 0.0; in_pos = False
        vals[j] = cash + shares * ci
    if in_pos and shares > 0:
        vals[-1] = shares * c[idx[-1]] * (1 - fee)
    return pd.Series(vals, index=pd.to_datetime(dt[idx])), trades

def simulate_stock(df, start, end, strat, params, capital, config, record=True):
    return sim_core(signal_arrays(df, strat, params, config), start, end, capital, config.fee_rate, config.stop_loss_pct, record=record)

def equity_stats(eq):
    if eq is None or len(eq) < 5: return None
    dr = eq.pct_change().dropna()
    if dr.std() < 1e-12: return None
    return (eq.iloc[-1] / eq.iloc[0] - 1) * 100, np.sqrt(252) * dr.mean() / dr.std()

# ==========================================================================
# 6. PERFORMANCE METRICS
# ==========================================================================
def metrics(equity, label=""):
    v = pd.Series(equity).dropna(); dr = v.pct_change().dropna()
    if len(v) < 2:
        return dict(label=label, v_start=0.0, v_end=0.0, total=0.0, cagr=0.0, sharpe=0.0, sortino=0.0, mdd=0.0, calmar=0.0, win=0.0)
    tot = (v.iloc[-1] / v.iloc[0] - 1) * 100
    ny = (v.index[-1] - v.index[0]).days / 365.25
    cagr = ((v.iloc[-1] / v.iloc[0]) ** (1 / ny) - 1) * 100 if ny > 0 else 0
    shp = np.sqrt(252) * dr.mean() / dr.std() if dr.std() > 1e-12 else 0.0
    neg = dr[dr < 0]
    sor = np.sqrt(252) * dr.mean() / neg.std() if len(neg) > 1 and neg.std() > 1e-12 else 0.0
    mdd = ((v - v.cummax()) / v.cummax()).min() * 100
    cal = cagr / abs(mdd) if mdd != 0 else 0.0
    win = (dr > 0).mean() * 100 if len(dr) else 0.0
    return dict(label=label, v_start=float(v.iloc[0]), v_end=float(v.iloc[-1]),
                total=tot, cagr=cagr, sharpe=shp, sortino=sor, mdd=mdd, calmar=cal, win=win)

# ==========================================================================
# 7. CALENDAR & LOOKBACK
# ==========================================================================
def build_calendar(ticker_dfs):
    return np.array(sorted(set().union(*[set(d["Date"]) for d in ticker_dfs.values()])))

def rebalance_dates(all_dates, config):
    invest = [d for d in all_dates if pd.Timestamp(d).year in config.invest_years]
    reb = []
    for y in config.invest_years:
        for q in config.rebalance_months:
            after = [d for d in invest if pd.Timestamp(d) >= pd.Timestamp(y, q, 1)]
            if after: reb.append(pd.Timestamp(after[0]))
    return sorted(set(reb))

def lookback_window(all_dates, d, config, n=None):
    if n is None:
        n = config.lookback_sess
    arr = pd.to_datetime(all_dates); idx = int(np.searchsorted(arr, pd.Timestamp(d)))
    return pd.Timestamp(arr[max(0, idx - n)]), pd.Timestamp(d)

# ==========================================================================
# 8. SCREENING AND WEIGHTING
# ==========================================================================
def screen(screen_cache, lb_start, lb_end, config):
    scores = {}
    for t, arrays in screen_cache.items():
        eq, _ = sim_core(arrays, lb_start, lb_end, 1e8, config.fee_rate, config.stop_loss_pct)
        st = equity_stats(eq)
        if st is None: continue
        ret, shp = st
        scores[t] = config.score_w_sharpe * shp + config.score_w_return * (ret / 100)
    return sorted(scores, key=lambda x: -scores[x])[:config.num_stocks]

def inverse_vol_weights(tickers, ticker_dfs, lb_start, lb_end):
    inv = {}
    for t in tickers:
        d = ticker_dfs[t]
        dr = d[(d["Date"] >= lb_start) & (d["Date"] < lb_end)]["Close"].pct_change().dropna()
        s = dr.std(); inv[t] = 1.0 / (s if s and s > 1e-9 else 0.05)
    tot = sum(inv.values())
    return {t: inv[t] / tot for t in tickers}

def portfolio_weights(tickers, ticker_dfs, all_dates, d, config):
    if config.weight_scheme == "equal" or len(tickers) == 0:
        return {t: 1.0 / len(tickers) for t in tickers}
    vs, _ = lookback_window(all_dates, d, config, config.vol_lookback)
    return inverse_vol_weights(tickers, ticker_dfs, vs, d)

def quarterly_subreturns(eq):
    s = pd.Series(eq).dropna().sort_index(); rets, labs = [], []
    for (y, q), g in s.groupby([s.index.year, s.index.quarter]):
        if len(g) >= 2:
            rets.append((g.iloc[-1] / g.iloc[0] - 1) * 100); labs.append(f"{str(y)[2:]}-Q{q}")
    return rets, labs

# ==========================================================================
# 9. PSO OPTIMIZATION
# ==========================================================================
def pso_optimize(selected, trimmed, lb_start, lb_end, strat, config):
    def objective(params):
        sh = []
        for t in selected:
            eq, tr = simulate_stock(trimmed[t], lb_start, lb_end, strat, params, 1e8, config, record=False)
            st = equity_stats(eq)
            sh.append((st[1] - 0.0) if st else -1.0)
        return -float(np.mean(sh))
    par = parametrization(strat)
    try: par.random_state.seed(config.random_seed)
    except Exception: pass
    rec = ng.optimizers.PSO(parametrization=par, budget=config.pso_budget).minimize(objective)
    return clean_params(strat, rec.value)

# ==========================================================================
# 10. RUN STRATEGY ORCHESTRATION
# ==========================================================================
def run_strategy(ticker_dfs, all_dates, strat, config, optimize=True, progress_bar=None, progress_val=0.0, step_val=0.0):
    reb = rebalance_dates(all_dates, config)
    if not reb:
        raise ValueError("Không tìm thấy ngày tái cơ cấu danh mục!")
    last_day = pd.Timestamp(max(all_dates)) + pd.Timedelta(days=1)
    
    dp = default_params(strat)
    screen_cache = {t: signal_arrays(df, strat, dp, config) for t, df in ticker_dfs.items()}

    capital = float(config.initial_capital)
    hist, q_log, q_returns, trades_all = [], [], [], []
    n_periods = len(reb)
    
    for i, d in enumerate(reb):
        if progress_bar is not None:
            current_p = progress_val + (i / n_periods) * step_val
            progress_bar.progress(min(current_p, 1.0), text=f"Đang chạy {strat} - Kỳ {i+1}/{n_periods} ({d.date()})...")
            
        q_end = reb[i + 1] if i + 1 < len(reb) else last_day
        lb_start, lb_end = lookback_window(all_dates, d, config)
        top = screen(screen_cache, lb_start, lb_end, config)
        if not top:
            continue
            
        if optimize:
            buf = lb_start - pd.Timedelta(days=160)
            trimmed = {t: ticker_dfs[t][(ticker_dfs[t]["Date"] >= buf) & (ticker_dfs[t]["Date"] < q_end)] for t in top}
            params = pso_optimize(top, trimmed, lb_start, lb_end, strat, config)
        else:
            params = dp
            
        weights = portfolio_weights(top, ticker_dfs, all_dates, d, config)

        start_cap, stock_eq = capital, {}
        for t in top:
            eq, tr = simulate_stock(ticker_dfs[t], d, q_end, strat, params, capital * weights[t], config, record=True)
            if eq is not None and len(eq): stock_eq[t] = eq
            for x in tr: x["Ticker"] = t
            trades_all.extend(tr)
            
        if not stock_eq:
            continue
            
        port = pd.DataFrame(stock_eq).sum(axis=1)
        idle = capital - sum(capital * weights[t] for t in stock_eq)
        port = port + idle
        
        for dt, val in port.items(): 
            hist.append({"Date": dt, "V": float(val)})
            
        capital = float(port.iloc[-1])
        ret = (capital / start_cap - 1) * 100
        q_log.append(dict(date=str(d.date()), top=top, params=params, ret=ret,
                          start=start_cap, end=capital,
                          weights={t: round(weights[t] * 100, 1) for t in top}))

    eq = pd.DataFrame(hist).set_index("Date")["V"].sort_index()
    eq = eq[~eq.index.duplicated(keep="last")]
    q_returns, q_labels = quarterly_subreturns(eq)
    buys = sum(1 for x in trades_all if x["Type"] == "MUA")
    ssig = sum(1 for x in trades_all if x["Type"] == "BAN" and x.get("Reason") == "TIN HIEU")
    ssl  = sum(1 for x in trades_all if x["Type"] == "BAN" and x.get("Reason") == "STOP-LOSS")
    
    return dict(equity=eq, metrics=metrics(eq, strat), q_log=q_log, q_returns=q_returns, q_labels=q_labels,
                n_buy=buys, n_sell_sig=ssig, n_sell_sl=ssl, trades=trades_all)

# ==========================================================================
# 11. BUY AND HOLD 100
# ==========================================================================
def buy_and_hold_100(ticker_dfs, all_dates, config):
    invest = sorted([pd.Timestamp(d) for d in all_dates if pd.Timestamp(d).year in config.invest_years])
    if not invest:
        raise ValueError("Không tìm thấy dữ liệu phù hợp trong năm đầu tư đã chọn!")
    first = invest[0]; alloc = config.initial_capital / len(ticker_dfs); holds = {}
    closes = {}
    for t, d in ticker_dfs.items():
        dd = d[d["Date"] >= first]
        bp = dd["Open"].iloc[0] if len(dd) else 0.0
        holds[t] = alloc * (1 - config.fee_rate) / bp if bp > 0 else 0.0
        closes[t] = d.set_index("Date")["Close"]
    vals = []
    for dt in invest:
        tot = sum(holds[t] * closes[t].get(dt, np.nan) for t in ticker_dfs if holds[t] > 0)
        vals.append({"Date": dt, "V": tot})
    eq = pd.DataFrame(vals).set_index("Date")["V"].sort_index().dropna()
    qr, qlab = quarterly_subreturns(eq)
    return dict(equity=eq, metrics=metrics(eq, "bnh"), q_returns=qr, q_labels=qlab)

# ==========================================================================
# 12. STATISTICAL TESTS
# ==========================================================================
def mean_ttest(data, mu=0.0):
    d = np.asarray([x for x in data if np.isfinite(x)], float)
    if len(d) < 3: 
        return dict(pvalue=None, significant=None, mean=(float(d.mean()) if len(d) else None), n=len(d))
    stat, pval = ttest_1samp(d, mu, alternative="greater")
    return dict(pvalue=float(pval), significant=bool(pval < 0.1), mean=float(d.mean()), n=len(d))

def wilcoxon_greater(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); n = min(len(a), len(b)); a, b = a[:n], b[:n]
    if n < 3 or np.allclose(a, b): 
        return dict(pvalue=None, significant=None, n=n)
    try:
        stat, pval = wilcoxon(a, b, alternative="greater")
        return dict(pvalue=float(pval), significant=bool(pval < 0.1), n=n)
    except Exception:
        return dict(pvalue=None, significant=None, n=n)

# ==========================================================================
# STREAMLIT UI LAYOUT
# ==========================================================================
st.markdown("<div class='main-title'>📈 Tối Ưu Hóa Danh Mục Đầu Tư Chứng Khoán HOSE</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Công cụ phân bổ tỷ trọng đầu tư & Backtest tự động sử dụng thuật toán PSO</div>", unsafe_allow_html=True)

# SIDEBAR CONFIGURATIONS
st.sidebar.header("📁 Dữ Liệu & Phạm Vi")

# Check for default files
import os
default_paths = [
    r"C:\Users\Tram Pham\Downloads\HOSE_2020_2023.csv",
    "HOSE_2020_2023.csv",
    "../HOSE_2020_2023.csv"
]
default_found = None
for dp in default_paths:
    if os.path.exists(dp):
        default_found = dp
        break

uploaded_file = st.sidebar.file_uploader("Tải lên file dữ liệu HOSE (CSV)", type=["csv"])

file_to_load = None
if uploaded_file is not None:
    file_to_load = uploaded_file
elif default_found is not None:
    file_to_load = default_found
    st.sidebar.info(f"Đang sử dụng dữ liệu mặc định tìm thấy tại:\n`{default_found}`")
else:
    st.sidebar.warning("Vui lòng tải lên một tệp CSV dữ liệu HOSE để bắt đầu.")

if file_to_load is not None:
    try:
        # Load raw data to identify years and tickers
        df_raw, use_adj = load_data(file_to_load)
        all_years = sorted(df_raw["Date"].dt.year.unique())
        
        st.sidebar.success(f"Tải thành công: {df_raw['Ticker'].nunique()} mã cổ phiếu, từ {df_raw['Date'].min().date()} đến {df_raw['Date'].max().date()}")
        if use_adj:
            st.sidebar.caption("💡 Hệ thống tự động sử dụng Giá Điều Chỉnh (`adj_open`, `adj_close`).")
        else:
            st.sidebar.caption("💡 Sử dụng Giá Thường (`open`, `close`) do không có cột điều chỉnh.")
            
        # Select Investment Years
        st.sidebar.subheader("📅 Khoảng Thời Gian")
        default_invest = [y for y in all_years if y > all_years[0]]
        if not default_invest:
            default_invest = [all_years[-1]]
            
        invest_years = st.sidebar.multiselect(
            "Năm backtest đầu tư",
            options=all_years,
            default=default_invest
        )
        
        observation_years = [y for y in all_years if y not in invest_years]
        if observation_years:
            st.sidebar.caption(f"Năm quan sát / khởi tạo tham số: {', '.join(map(str, observation_years))}")
        
        # Universe Filtering
        st.sidebar.subheader("🎯 Vũ Trụ Đầu Tư")
        all_sectors = list(INDUSTRY_TICKERS.keys())
        selected_sectors = st.sidebar.multiselect(
            "Lọc theo nhóm ngành",
            options=all_sectors,
            default=[]
        )
        
        # Parameter Adjustments
        with st.sidebar.expander("⚙️ Tham Số Backtest & PSO", expanded=False):
            initial_capital = st.number_input("Vốn ban đầu (VND)", min_value=10_000_000, max_value=10_000_000_000_000, value=1_000_000_000, step=10_000_000)
            fee_rate = st.slider("Phí giao dịch (mỗi lệnh %)", min_value=0.0, max_value=2.0, value=0.15, step=0.01) / 100.0
            stop_loss = st.slider("Mức Cắt Lỗ (Stop-Loss %)", min_value=0, max_value=50, value=15, step=1) / 100.0
            num_stocks = st.slider("Số cổ phiếu nắm giữ mỗi kỳ (N)", min_value=1, max_value=20, value=5, step=1)
            pso_budget = st.slider("Số vòng lặp PSO (PSO Budget)", min_value=10, max_value=200, value=50, step=5)
            
            rebal_freq = st.selectbox(
                "Tần suất tái cơ cấu danh mục",
                options=["Hàng năm (Annual)", "Bán niên (Semi-Annual)", "Hàng quý (Quarterly)"],
                index=0
            )
            
            weight_scheme_str = st.selectbox(
                "Phương pháp phân bổ tỷ trọng",
                options=["Chia đều (Equal)", "Nghịch đảo biến động (Inverse Volatility)"],
                index=0
            )
            
            lookback_sess = st.number_input("Thời gian quan sát tối ưu (phiên)", min_value=50, max_value=500, value=252, step=10)
            vol_lookback = st.number_input("Thời gian tính biến động (phiên)", min_value=10, max_value=200, value=63, step=5)
            trend_window = st.number_input("Thời gian lọc xu hướng SMA (phiên)", min_value=10, max_value=300, value=100, step=5)
            
        # Mapping rebalance month
        rebal_map = {
            "Hàng năm (Annual)": [1],
            "Bán niên (Semi-Annual)": [1, 7],
            "Hàng quý (Quarterly)": [1, 4, 7, 10]
        }
        rebalance_months = rebal_map[rebal_freq]
        
        weight_map = {
            "Chia đều (Equal)": "equal",
            "Nghịch đảo biến động (Inverse Volatility)": "invvol"
        }
        weight_scheme = weight_map[weight_scheme_str]
        
        # Build config object
        config = Config(
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            stop_loss_pct=stop_loss,
            num_stocks=num_stocks,
            pso_budget=pso_budget,
            lookback_sess=lookback_sess,
            vol_lookback=vol_lookback,
            trend_window=trend_window,
            random_seed=42,
            score_w_sharpe=0.60,
            score_w_return=0.40,
            invest_years=invest_years,
            rebalance_months=rebalance_months,
            weight_scheme=weight_scheme
        )
        
        # Filter ticker list based on selected industries
        eligible_tickers = set()
        if selected_sectors:
            for s in selected_sectors:
                eligible_tickers.update(INDUSTRY_TICKERS[s])
            # Filter df_raw
            df_filtered = df_raw[df_raw["Ticker"].isin(eligible_tickers)].reset_index(drop=True)
            if df_filtered.empty:
                st.error("Không có cổ phiếu nào của ngành đã chọn nằm trong file dữ liệu. Vui lòng chọn ngành khác hoặc bỏ chọn lọc ngành.")
                st.stop()
            st.sidebar.info(f"Đã lọc: Còn {df_filtered['Ticker'].nunique()} mã thuộc các ngành đã chọn.")
        else:
            df_filtered = df_raw

        # RUN ENGINE BUTTON
        run_btn = st.sidebar.button("🚀 Bắt đầu Backtest & Tối Ưu Hóa", use_container_width=True)

        if run_btn or "backtest_results" in st.session_state:
            # Run simulation if button clicked or already saved in state
            if run_btn or ("config_params" not in st.session_state or st.session_state["config_params"] != str(config.__dict__) + str(selected_sectors)):
                
                # Check for investment years selected
                if not invest_years:
                    st.error("Vui lòng chọn ít nhất một năm đầu tư trong sidebar.")
                    st.stop()
                    
                # Create dictionary of ticker dataframes
                ticker_dfs = {t: df_filtered[df_filtered["Ticker"] == t].copy() for t in sorted(df_filtered["Ticker"].unique())}
                all_dates = build_calendar(ticker_dfs)
                
                # Create progress bar
                progress_bar = st.progress(0.0, text="Đang bắt đầu backtest...")
                
                # Run strategies
                results = {}
                try:
                    # 1. B&H
                    progress_bar.progress(0.05, text="Đang mô phỏng chiến lược Mua & Nắm giữ (B&H)...")
                    results["bnh"] = buy_and_hold_100(ticker_dfs, all_dates, config)
                    
                    # 2. RSI Pure (Optimized)
                    results["rsi"] = run_strategy(ticker_dfs, all_dates, "rsi", config, optimize=True, 
                                                  progress_bar=progress_bar, progress_val=0.10, step_val=0.20)
                    
                    # 3. RSI + MACD (Optimized)
                    results["rsi_macd"] = run_strategy(ticker_dfs, all_dates, "rsi_macd", config, optimize=True, 
                                                       progress_bar=progress_bar, progress_val=0.30, step_val=0.25)
                    
                    # 4. RSI + MACD Default (No PSO)
                    results["rsi_macd_default"] = run_strategy(ticker_dfs, all_dates, "rsi_macd", config, optimize=False, 
                                                               progress_bar=progress_bar, progress_val=0.55, step_val=0.15)
                    
                    # 5. Bollinger + RSI (Optimized)
                    results["bollinger_rsi"] = run_strategy(ticker_dfs, all_dates, "bollinger_rsi", config, optimize=True, 
                                                            progress_bar=progress_bar, progress_val=0.70, step_val=0.25)
                    
                    # Statistical verification calculations
                    progress_bar.progress(0.95, text="Đang phân tích kiểm định thống kê và tổng hợp kết quả...")
                    
                    best_strat = max(["rsi", "rsi_macd", "bollinger_rsi"], key=lambda k: results[k]["metrics"]["sharpe"])
                    
                    st.session_state["backtest_results"] = results
                    st.session_state["best_strategy"] = best_strat
                    st.session_state["config_params"] = str(config.__dict__) + str(selected_sectors)
                    st.session_state["ticker_dfs"] = ticker_dfs
                    
                    progress_bar.progress(1.0, text="Hoàn tất!")
                    st.balloons()
                    progress_bar.empty()
                    
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"Đã xảy ra lỗi khi chạy backtest: {str(e)}")
                    st.exception(e)
                    st.stop()
            
            # Load from session state
            results = st.session_state["backtest_results"]
            best_strat = st.session_state["best_strategy"]
            ticker_dfs = st.session_state["ticker_dfs"]
            
            strategy_names = {
                "rsi": "RSI Thuần (PSO)",
                "rsi_macd": "RSI + MACD (PSO)",
                "bollinger_rsi": "RSI + Bollinger (PSO)",
                "rsi_macd_default": "RSI + MACD (Mặc định)",
                "bnh": "Mua & Nắm giữ (B&H)"
            }
            
            # ----------------------------------------------------
            # DASHBOARD - KPI CARDS
            # ----------------------------------------------------
            st.markdown("### 📊 Chỉ Số Hiệu Suất Tổng Quan")
            cols = st.columns(4)
            
            dashboard_keys = ["rsi_macd", "bollinger_rsi", "rsi", "bnh"]
            
            for idx, key in enumerate(dashboard_keys):
                res = results[key]
                met = res["metrics"]
                is_best = (key == best_strat)
                
                with cols[idx]:
                    card_class = "metric-card-best" if is_best else "metric-card"
                    val_class = "metric-val-best" if is_best else "metric-val"
                    best_tag = "<span class='tag-best'>TỐT NHẤT (Sharpe)</span><br>" if is_best else ""
                    
                    st.markdown(f"""
                    <div class='{card_class}'>
                        {best_tag}
                        <div style='font-size: 16px; font-weight: bold; color: { '#1E3A8A' if is_best else '#4B5563' }; margin-bottom: 8px;'>{strategy_names[key]}</div>
                        <div class='metric-title'>Tỷ suất sinh lời tổng</div>
                        <div class='{val_class}' style='color: {'#059669' if met['total'] >= 0 else '#DC2626'}'>{met['total']:+.2f}%</div>
                        <div class='metric-title' style='margin-top:8px;'>CAGR (%/năm)</div>
                        <div class='{val_class}'>{met['cagr']:.2f}%</div>
                        <div class='metric-title' style='margin-top:8px;'>Sharpe Ratio</div>
                        <div class='{val_class}'>{met['sharpe']:.3f}</div>
                        <div class='metric-title' style='margin-top:8px;'>Max Drawdown</div>
                        <div class='{val_class}' style='color: #DC2626;'>{met['mdd']:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # ----------------------------------------------------
            # TABS VIEW
            # ----------------------------------------------------
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Biểu Đồ Hiệu Suất", 
                "🔄 Nhật Ký Tái Cơ Cấu", 
                "📜 Nhật Ký Giao Dịch Chi Tiết", 
                "🔬 Kiểm Định Thống Kê",
                "🔍 Khám Phá Dữ Liệu"
            ])
            
            # TAB 1: CHARTS
            with tab1:
                st.markdown("### 📊 Biểu đồ so sánh hiệu suất (2021-2023)")
                
                col1, col2 = st.columns([7, 3])
                
                with col1:
                    # Chart 1: Equity Curve
                    fig1, ax1 = plt.subplots(figsize=(12, 5.5))
                    C = {"rsi": "#888780", "rsi_macd": "#185FA5", "bollinger_rsi": "#0F6E56", "bnh": "#993C1D", "rsi_macd_default": "#F59E0B"}
                    
                    for k in ["rsi", "rsi_macd", "bollinger_rsi", "rsi_macd_default", "bnh"]:
                        eq = results[k]["equity"]
                        ax1.plot(eq.index, eq / 1e9, color=C[k], lw=2.8 if k == best_strat else 1.8, 
                                 ls="--" if k in ["bnh", "rsi_macd_default"] else "-",
                                 label=f"{strategy_names[k]} ({results[k]['metrics']['total']:+.1f}%)")
                                 
                    # Draw vertical lines for rebalance dates
                    for r in results[best_strat]["q_log"]:
                        ax1.axvline(pd.Timestamp(r["date"]), color="gray", lw=0.6, ls=":", alpha=0.5)
                        
                    ax1.axhline(config.initial_capital / 1e9, color="gray", lw=1.0, ls="--", alpha=0.7)
                    ax1.set_title("Biểu đồ 1: Giá trị tài sản danh mục (Tỷ VND)", fontweight="bold", fontsize=12)
                    ax1.set_ylabel("Giá trị tài sản (Tỷ VND)")
                    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
                    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30)
                    ax1.legend(loc="upper left", fontsize=9)
                    ax1.grid(True, alpha=0.2)
                    ax1.spines['top'].set_visible(False)
                    ax1.spines['right'].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig1)
                    plt.close()
                    
                with col2:
                    # Chart 3: Metric Comparison Bar Chart
                    fig3, ax3 = plt.subplots(figsize=(5.5, 5.2))
                    L = ["Tổng LN\n(%)", "CAGR\n(%/năm)", "Sharpe\n(x10)", "Calmar\n(x10)"]
                    
                    def get_metrics_vector(k):
                        return [
                            results[k]["metrics"]["total"], 
                            results[k]["metrics"]["cagr"],
                            results[k]["metrics"]["sharpe"] * 10, 
                            results[k]["metrics"]["calmar"] * 10
                        ]
                        
                    x = np.arange(len(L))
                    width = 0.15
                    
                    for idx, k in enumerate(["rsi", "rsi_macd", "bollinger_rsi", "bnh"]):
                        ax3.bar(x + (idx - 1.5) * width, get_metrics_vector(k), width, color=C[k], label=strategy_names[k])
                        
                    ax3.set_xticks(x)
                    ax3.set_xticklabels(L, fontsize=9)
                    ax3.axhline(0, color="black", lw=0.8)
                    ax3.legend(fontsize=8, loc="upper right")
                    ax3.grid(axis="y", alpha=0.2)
                    ax3.set_title("So sánh các chỉ số tài chính", fontweight="bold", fontsize=11)
                    ax3.spines['top'].set_visible(False)
                    ax3.spines['right'].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig3)
                    plt.close()
                
                # Chart 2: Quarterly Return Bar Chart of the Best Strategy
                st.markdown(f"#### 📅 Lợi nhuận từng quý của Chiến lược Tốt Nhất ({strategy_names[best_strat]})")
                fig2, ax2 = plt.subplots(figsize=(15, 3.8))
                
                labels = results[best_strat].get("q_labels") or [r["date"][2:7] for r in results[best_strat]["q_log"]]
                rts = results[best_strat]["q_returns"]
                
                if len(rts) > 0:
                    bars = ax2.bar(labels, rts, color=[C[best_strat] if r >= 0 else "#A32D2D" for r in rts], width=0.55, edgecolor="white")
                    for bar in bars:
                        height = bar.get_height()
                        ax2.annotate(f"{height:+.1f}%", 
                                     (bar.get_x() + bar.get_width() / 2, height),
                                     xytext=(0, 4 if height >= 0 else -14), 
                                     textcoords="offset points",
                                     ha="center", fontsize=8.5, fontweight="bold")
                                     
                    ax2.axhline(0, color="black", lw=0.8)
                    ax2.set_title(f"Biểu đồ 2: Tỷ suất sinh lời theo Quý dương lịch – Chiến lược {strategy_names[best_strat]}", fontweight="bold", fontsize=11)
                    ax2.set_ylabel("Lợi nhuận (%)")
                    ax2.grid(axis="y", alpha=0.2)
                    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30)
                    ax2.spines['top'].set_visible(False)
                    ax2.spines['right'].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close()
                else:
                    st.warning("Không đủ dữ liệu lịch để tính toán tỷ suất sinh lời từng quý.")
            
            # TAB 2: REBALANCING LOG
            with tab2:
                st.markdown("### 🔄 Nhật Ký Tái Cơ Cấu & Tối Ưu Tham Số")
                
                # Choose strategy to view
                view_strat = st.selectbox(
                    "Chọn chiến lược để xem lịch sử tái cơ cấu:",
                    options=["rsi_macd", "bollinger_rsi", "rsi", "rsi_macd_default"],
                    format_func=lambda x: strategy_names[x]
                )
                
                q_log = results[view_strat]["q_log"]
                
                if q_log:
                    # Build summary table of rebalancing dates
                    periods_data = []
                    for idx, log in enumerate(q_log):
                        # Construct parameters string
                        param_str = ", ".join([f"{k}={v}" for k, v in log["params"].items()])
                        # Construct weights string
                        weight_str = ", ".join([f"{t}: {w}%" for t, w in log["weights"].items()])
                        
                        periods_data.append({
                            "Kỳ": idx + 1,
                            "Ngày tái cơ cấu": log["date"],
                            "Top Cổ Phiếu Chọn": ", ".join(log["top"]),
                            "Tỷ Trọng Phân Bổ": weight_str,
                            "Tham Số Tối Ưu": param_str,
                            "Vốn Đầu Kỳ": f"{log['start']:,.0f} VND",
                            "Vốn Cuối Kỳ": f"{log['end']:,.0f} VND",
                            "TSSL Kỳ (%)": f"{log['ret']:+.2f}%"
                        })
                        
                    df_periods = pd.DataFrame(periods_data)
                    st.dataframe(df_periods, use_container_width=True, hide_index=True)
                else:
                    st.warning("Không có dữ liệu nhật ký tái cơ cấu.")
            
            # TAB 3: TRANSACTION LOG
            with tab3:
                st.markdown("### 📜 Nhật Ký Lệnh Giao Dịch Chi Tiết")
                
                # Select strategy
                tx_strat = st.selectbox(
                    "Chọn chiến lược xem lịch sử giao dịch:",
                    options=["rsi_macd", "bollinger_rsi", "rsi", "rsi_macd_default"],
                    format_func=lambda x: strategy_names[x],
                    key="tx_strat_select"
                )
                
                trades = results[tx_strat]["trades"]
                
                if trades:
                    # Convert to dataframe
                    df_trades = pd.DataFrame(trades)
                    
                    # Convert Datetime objects to string
                    df_trades["Date"] = pd.to_datetime(df_trades["Date"]).dt.strftime("%Y-%m-%d")
                    
                    # Formatting values
                    df_trades_styled = df_trades.copy()
                    df_trades_styled["Price"] = df_trades_styled["Price"].map(lambda x: f"{x:,.0f} VND" if pd.notnull(x) else "")
                    
                    # PnL only exists for sell trades
                    if "PnL" in df_trades_styled.columns:
                        df_trades_styled["PnL"] = df_trades_styled["PnL"].map(lambda x: f"{x:+,.0f} VND" if pd.notnull(x) else "")
                        
                    # Reorder columns
                    cols_order = ["Date", "Ticker", "Type", "Price"]
                    if "PnL" in df_trades.columns:
                        cols_order.append("PnL")
                    if "Reason" in df_trades.columns:
                        cols_order.append("Reason")
                        
                    df_trades_styled = df_trades_styled[cols_order]
                    
                    # Filtering widgets
                    cols_filter = st.columns(3)
                    with cols_filter[0]:
                        filter_ticker = st.multiselect("Lọc theo Mã cổ phiếu:", options=sorted(df_trades["Ticker"].unique()))
                    with cols_filter[1]:
                        filter_type = st.multiselect("Lọc Loại giao dịch:", options=["MUA", "BAN"])
                    with cols_filter[2]:
                        if "Reason" in df_trades.columns:
                            filter_reason = st.multiselect("Lọc Lý do bán:", options=list(df_trades["Reason"].dropna().unique()))
                        else:
                            filter_reason = []
                            
                    # Apply filters
                    df_filtered_trades = df_trades.copy()
                    if filter_ticker:
                        df_filtered_trades = df_filtered_trades[df_filtered_trades["Ticker"].isin(filter_ticker)]
                    if filter_type:
                        df_filtered_trades = df_filtered_trades[df_filtered_trades["Type"].isin(filter_type)]
                    if filter_reason and "Reason" in df_filtered_trades.columns:
                        df_filtered_trades = df_filtered_trades[df_filtered_trades["Reason"].isin(filter_reason)]
                        
                    # Format filtered df for display
                    df_display = df_filtered_trades.copy()
                    df_display["Price"] = df_display["Price"].map(lambda x: f"{x:,.0f} VND" if pd.notnull(x) else "")
                    if "PnL" in df_display.columns:
                        df_display["PnL"] = df_display["PnL"].map(lambda x: f"{x:+,.0f} VND" if pd.notnull(x) else "")
                    df_display = df_display[cols_order]
                    
                    st.markdown(f"**Số lượng giao dịch tìm thấy:** {len(df_filtered_trades)} lệnh")
                    st.dataframe(df_display, use_container_width=True, hide_index=True)
                    
                    # CSV Download Button
                    csv_buffer = io.StringIO()
                    df_filtered_trades.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
                    st.download_button(
                        label="📥 Tải xuống Nhật ký giao dịch (CSV)",
                        data=csv_buffer.getvalue(),
                        file_name=f"nhat_ky_giao_dich_{tx_strat}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("Không phát sinh lệnh giao dịch nào trong quá trình backtest.")
            
            # TAB 4: STATISTICAL TESTS
            with tab4:
                st.markdown("### 🔬 Kiểm Định Ý Nghĩa Thống Kê")
                st.markdown("""
                Chúng tôi áp dụng các kiểm định thống kê trên **lợi nhuận theo quý (OOS)** của các chiến lược để đảm bảo rằng hiệu suất vượt trội không phải do ngẫu nhiên:
                1. **T-test 1 mẫu**: Kiểm định xem trung bình lợi nhuận quý Out-of-Sample có thực sự lớn hơn 0 một cách có ý nghĩa thống kê hay không ($H_0: \mu \le 0$; $H_1: \mu > 0$).
                2. **Wilcoxon Signed-Rank Test (Toán tử lớn hơn)**: 
                   - So sánh lợi nhuận quý của chiến lược **Tối ưu hóa (PSO)** so với phiên bản **Mặc định (Không tối ưu)**.
                   - So sánh lợi nhuận quý của chiến lược **Tối ưu hóa** so với chiến lược chuẩn **Mua & Nắm giữ (B&H)**.
                """)
                
                # Perform testing calculations
                ttest_res = mean_ttest(results["rsi_macd"]["q_returns"], 0.0)
                wilcoxon_opt = wilcoxon_greater(results["rsi_macd"]["q_returns"], results["rsi_macd_default"]["q_returns"])
                wilcoxon_bnh = wilcoxon_greater(results["rsi_macd"]["q_returns"], results["bnh"]["q_returns"])
                
                col_st1, col_st2, col_st3 = st.columns(3)
                
                with col_st1:
                    st.subheader("1. Kiểm định T-test (RSI + MACD PSO)")
                    st.metric(label="Trung bình Lợi nhuận Quý OOS", value=f"{ttest_res['mean']:.2f}%" if ttest_res['mean'] is not None else "N/A")
                    st.metric(label="Trị số p (p-value)", value=f"{ttest_res['pvalue']:.4f}" if ttest_res['pvalue'] is not None else "N/A")
                    if ttest_res['significant'] is not None:
                        status = "✅ Có ý nghĩa thống kê (p < 0.1)" if ttest_res['significant'] else "❌ Chưa có ý nghĩa thống kê (p >= 0.1)"
                        st.info(status)
                    else:
                        st.warning("Không đủ số lượng mẫu (quý) để thực hiện t-test.")
                        
                with col_st2:
                    st.subheader("2. Kiểm định Wilcoxon: PSO vs Mặc định")
                    st.metric(label="Hiệu số PnL tổng (PSO - Mặc định)", value=f"{results['rsi_macd']['metrics']['total'] - results['rsi_macd_default']['metrics']['total']:+.2f}%")
                    st.metric(label="Trị số p (p-value)", value=f"{wilcoxon_opt['pvalue']:.4f}" if wilcoxon_opt['pvalue'] is not None else "N/A")
                    if wilcoxon_opt['significant'] is not None:
                        status = "✅ Có ý nghĩa thống kê (p < 0.1)" if wilcoxon_opt['significant'] else "❌ Chưa có ý nghĩa thống kê (p >= 0.1)"
                        st.info(status)
                    else:
                        st.warning("Không đủ số lượng mẫu hoặc dữ liệu bằng nhau.")
                        
                with col_st3:
                    st.subheader("3. Kiểm định Wilcoxon: PSO vs B&H")
                    st.metric(label="Hiệu số PnL tổng (PSO - B&H)", value=f"{results['rsi_macd']['metrics']['total'] - results['bnh']['metrics']['total']:+.2f}%")
                    st.metric(label="Trị số p (p-value)", value=f"{wilcoxon_bnh['pvalue']:.4f}" if wilcoxon_bnh['pvalue'] is not None else "N/A")
                    if wilcoxon_bnh['significant'] is not None:
                        status = "✅ Có ý nghĩa thống kê (p < 0.1)" if wilcoxon_bnh['significant'] else "❌ Chưa có ý nghĩa thống kê (p >= 0.1)"
                        st.info(status)
                    else:
                        st.warning("Không đủ số lượng mẫu hoặc dữ liệu bằng nhau.")
                        
            # TAB 5: DATA EXPLORER
            with tab5:
                st.markdown("### 🔍 Khám Phá Dữ Liệu Tải Lên")
                
                col_ex1, col_ex2 = st.columns([3, 7])
                
                with col_ex1:
                    st.write("**Thông tin thống kê tổng quan:**")
                    st.write(f"- **Tổng số mã cổ phiếu:** {df_filtered['Ticker'].nunique()}")
                    st.write(f"- **Tổng số dòng dữ liệu:** {len(df_filtered):,}")
                    st.write(f"- **Ngày giao dịch đầu tiên:** {df_filtered['Date'].min().strftime('%Y-%m-%d')}")
                    st.write(f"- **Ngày giao dịch cuối cùng:** {df_filtered['Date'].max().strftime('%Y-%m-%d')}")
                    
                    st.write("**Thống kê số lượng mã giao dịch theo năm:**")
                    df_filtered["Year"] = df_filtered["Date"].dt.year
                    yearly_counts = df_filtered.groupby("Year")["Ticker"].nunique()
                    st.table(yearly_counts)
                    
                with col_ex2:
                    st.write("**Xem trước dữ liệu thô (20 dòng đầu tiên):**")
                    st.dataframe(df_filtered.head(20), use_container_width=True, hide_index=True)
                    
                    st.write("**Danh sách các mã cổ phiếu trong vũ trụ đã chọn:**")
                    tickers_str = ", ".join(sorted(df_filtered["Ticker"].unique()))
                    st.caption(tickers_str)
                    
    except Exception as e:
        st.error(f"Lỗi khi đọc file dữ liệu: {str(e)}")
        st.exception(e)
else:
    st.info("💡 Bạn có thể tải file dữ liệu HOSE dạng CSV lên bằng sidebar bên trái để thực hiện tối ưu hóa danh mục.")
    st.markdown("""
    ### 📂 Định dạng dữ liệu đầu vào yêu cầu:
    Tệp dữ liệu CSV tải lên cần có các cột sau:
    1. `ticker` (hoặc `Ticker`): Mã chứng khoán (ví dụ: HPG, VCB, VIC,...)
    2. `date` (hoặc `Date`): Ngày giao dịch dạng `MM/DD/YYYY` hoặc `YYYY-MM-DD`
    3. `open` (hoặc `adj_open`): Giá mở cửa (hoặc giá mở cửa điều chỉnh)
    4. `close` (hoặc `adj_close`): Giá đóng cửa (hoặc giá đóng cửa điều chỉnh)
    
    *Hệ thống sẽ tự động ưu tiên sử dụng Giá điều chỉnh (`adj_open`, `adj_close`) nếu phát hiện chúng trong file dữ liệu của bạn để đảm bảo kết quả backtest chính xác.*
    """)
