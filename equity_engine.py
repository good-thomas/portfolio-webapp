import math
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, request, jsonify
from flask_cors import CORS # <--- Das muss importiert sein

app = Flask(__name__)
CORS(app) # <--- Das erlaubt JEDER Website die Abfrage

# --- Konfiguration ---
START_DATE = "2000-01-01"
LOOKBACK_MONTHS = 12
DEFAULT_COST_RATE = 0.001
DEFAULT_SECTOR_PROXY_SET = "us_long_history"

GICS_BUCKETS_US_LONG_HISTORY = {
    "energy": "XLE", "materials": "XLB", "industrials": "XLI", "cons_disc": "XLY",
    "cons_staples": "XLP", "health_care": "XLV", "financials": "XLF", "technology": "XLK",
    "utilities": "XLU", "real_estate": "IYR", "semis": "SMH", "biotech": "IBB",
    "hc_equip": "IHI", "banks": "KBE", "aerospace_defense": "ITA", "infrastructure": "PAVE",
    "homebuilders": "XHB", "transportation": "IYT", "gold_miners": "GDX", "oil_services": "OIH",
    "metals_mining": "XME", "software": "IGV"
}

GICS_BUCKETS_UCITS = {
    "energy": "WNRG.DE", "materials": "XMWS.DE", "defense": "DFND.L", "industrials_ex": "WIND.DE",
    "cons_disc": "SC0G.DE", "cons_staples": "WCSS.DE", "pharma_bio": "BIOT.DE", "hc_equip": "IHI",
    "banks": "EXV1.DE", "fin_serv": "WFIN.DE", "software": "IGPT.DE", "semis": "VVSM.DE",
    "comm_serv": "XWTS.DE", "utilities": "WUTI.DE", "real_estate": "DPRE.DE"
}

SECTOR_PROXY_SETS = {"ucits": GICS_BUCKETS_UCITS, "us_long_history": GICS_BUCKETS_US_LONG_HISTORY}

# --- Hilfsfunktionen ---
def load_data(sector_proxy_set):
    buckets = SECTOR_PROXY_SETS.get(sector_proxy_set, GICS_BUCKETS_US_LONG_HISTORY)
    tickers = sorted(set(["ACWI"] + list(buckets.values())))
    data = yf.download(tickers, start=START_DATE, auto_adjust=True, progress=False)["Close"]
    
    series = {"equities": data["ACWI"].resample("ME").last()}
    for name, tkr in buckets.items():
        if tkr in data.columns:
            series[name] = data[tkr].resample("ME").last()
            
    prices = pd.concat(series, axis=1).ffill().dropna(subset=["equities"])
    return prices, prices.pct_change(), buckets

def compute_score(prices, asset, i):
    if i < 12: return np.nan
    p = prices[asset]
    curr = p.iloc[i]
    try:
        return 0.5*(curr/p.iloc[i-3]-1) + 0.3*(curr/p.iloc[i-6]-1) + 0.2*(curr/p.iloc[i-12]-1)
    except: return np.nan

def run_backtest(start_date=None, sector_proxy_set=DEFAULT_SECTOR_PROXY_SET, cost_rate=DEFAULT_COST_RATE):
    prices, rets, buckets = load_data(sector_proxy_set)
    start_i = np.where(prices.index >= pd.to_datetime(start_date))[0][0] if start_date else 12
    
    engine_rets = []
    acwi_rets = []
    dates = []
    weight_hist = []
    prev_w = None

    for i in range(start_i, len(prices)-1):
        # Einfache Logik: Top 3 Sektoren vs ACWI
        eq_score = compute_score(prices, "equities", i)
        b_scores = {b: compute_score(prices, b, i) for b in buckets.keys() if b in prices.columns}
        top = sorted([b for b, s in b_scores.items() if pd.notna(s) and s > eq_score], 
                     key=lambda x: b_scores[x], reverse=True)[:3]
        
        weights = {"equities": 1.0}
        if top:
            sec_w = 0.5 / len(top) # 50% Overlay
            weights = {"equities": 0.5, **{b: sec_w for b in top}}

        # Kosten & Performance
        turnover = sum(abs(weights.get(a, 0) - (prev_w.get(a, 0) if prev_w else 0)) for a in set(weights) | set(prev_w or {}))
        m_ret = sum(weights[a] * rets[a].iloc[i+1] for a in weights if a in rets.columns) - (turnover * cost_rate)
        
        engine_rets.append(m_ret)
        acwi_rets.append(rets["equities"].iloc[i+1])
        dates.append(prices.index[i+1])
        weight_hist.append({"date": prices.index[i+1].strftime("%Y-%m-%d"), "weights": weights, "selected": top})
        prev_w = weights

    # Matrix Berechnung
    df_m = pd.DataFrame({"ret": engine_rets}, index=dates)
    matrix = df_m.groupby([df_m.index.year, df_m.index.month])["ret"].sum().unstack()
    
    return {
        "series": {
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            "equity_engine": (1 + pd.Series(engine_rets)).cumprod().tolist(),
            "acwi": (1 + pd.Series(acwi_rets)).cumprod().tolist()
        },
        "performance": {
            "equity_engine": calc_stats(pd.Series(engine_rets, index=dates)),
            "acwi": calc_stats(pd.Series(acwi_rets, index=dates))
        },
        "matrix": matrix.to_dict(orient="index"),
# Korrigierte Zeile:
"selection_counts": pd.Series([b for h in weight_hist for b in h["selected"]]).value_counts().to_dict() if weight_hist else {},        "latest_weights": weight_hist[-1] if weight_hist else {}
    }

def calc_stats(r):
    curr = (1+r).cumprod()
    years = (r.index[-1] - r.index[0]).days / 365.25
    cagr = curr.iloc[-1]**(1/years)-1
    vola = r.std() * math.sqrt(12)
    return {"cagr": float(cagr), "vola": float(vola), "max_drawdown": float((curr/curr.cummax()-1).min()), "sharpe": float(cagr/vola), "total_return": float(curr.iloc[-1]-1)}

@app.route("/api/equity-engine")
def api():
    return jsonify(run_backtest(
        start_date=request.args.get("start_date"),
        sector_proxy_set=request.args.get("sector_proxy_set", "us_long_history"),
        cost_rate=float(request.args.get("cost_rate", 0.001))
    ))

if __name__ == "__main__":
    import os
    # Render weist automatisch einen Port über die Umgebungsvariable zu
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
