import math
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import traceback

app = Flask(__name__)
CORS(app)

# --- Hilfsfunktionen ---

def calc_stats(r):
    if r.empty or r.isnull().all():
        return {"cagr": 0, "vola": 0, "max_drawdown": 0, "sharpe": 0, "total_return": 0}
    curr = (1 + r).cumprod()
    days = (r.index[-1] - r.index[0]).days
    years = days / 365.25 if days > 0 else 1
    cagr = float(curr.iloc[-1]**(1/years) - 1) if years > 0 else 0
    vola = float(r.std() * math.sqrt(12))
    return {
        "cagr": round(cagr, 4), "vola": round(vola, 4), 
        "max_drawdown": round(float((curr / curr.cummax() - 1).min()), 4),
        "sharpe": round(cagr / vola, 4) if vola > 0 else 0,
        "total_return": round(float(curr.iloc[-1] - 1), 4)
    }

def compute_score(series, i, w1, w3, w6, w12):
    try:
        if i < 12 or i >= len(series): return -999
        p = series
        if pd.isna(p.iloc[i]) or pd.isna(p.iloc[i-12]): return -999
        r1 = (p.iloc[i]/p.iloc[i-1]-1)
        r3 = (p.iloc[i]/p.iloc[i-3]-1)
        r6 = (p.iloc[i]/p.iloc[i-6]-1)
        r12 = (p.iloc[i]/p.iloc[i-12]-1)
        return (w1 * r1) + (w3 * r3) + (w6 * r6) + (w12 * r12)
    except: return -999

# --- Optimierungs-Logik ---

def find_best_params(price_slice, sector_list, y_monthly_premium):
    # Die 5 Master-Settings inklusive deiner 6M/12M Optimierung
    configs = [
        {'w1': 0.0, 'w3': 0.0, 'w6': 0.2, 'w12': 0.8}, # Thomas' Empiric Choice
        {'w1': 0.0, 'w3': 0.0, 'w6': 0.5, 'w12': 0.5}, # Balanced Mix
        {'w1': 0.0, 'w3': 0.3, 'w6': 0.4, 'w12': 0.3}, # Quartals-Fokus
        {'w1': 0.2, 'w3': 0.3, 'w6': 0.3, 'w12': 0.2}, # Tactical/Reaktiv
        {'w1': 0.4, 'w3': 0.4, 'w6': 0.2, 'w12': 0.0}  # Mean Reversion / Rebound
    ]
    
    best_cfg = None
    max_sharpe = -1
    acwi_rets = price_slice["equities"].pct_change().dropna()
    acwi_stats = calc_stats(acwi_rets)
    target_cagr = acwi_stats["cagr"] + (y_monthly_premium * 12)

    # Simulation im Fenster (jeder 2. Monat für Speed)
    for cfg in configs:
        sim_rets = []
        for j in range(12, len(price_slice)-1, 2):
            acwi_s = compute_score(price_slice["equities"], j, **cfg)
            bil_s = compute_score(price_slice["cash"], j, **cfg)
            s_scores = {s: compute_score(price_slice[s], j, **cfg) for s in sector_list}
            qualified = [s for s, sc in s_scores.items() if sc > (acwi_s + y_monthly_premium) and sc > bil_s]
            
            if qualified:
                m_ret = price_slice[qualified[:3]].pct_change().iloc[j+1].mean()
            elif acwi_s > bil_s:
                m_ret = acwi_rets.iloc[j]
            else:
                m_ret = price_slice["cash"].pct_change().iloc[j+1]
            sim_rets.append(m_ret)
        
        if not sim_rets: continue
        res = calc_stats(pd.Series(sim_rets))
        if res["cagr"] > target_cagr and res["sharpe"] > acwi_stats["sharpe"]:
            if res["sharpe"] > max_sharpe:
                max_sharpe = res["sharpe"]
                best_cfg = cfg
    return best_cfg

# --- API & Backtest ---

TICKERS_V3 = {
    "equities": "ACWI", "cash": "BIL",
    "sectors": {
        "defense": "ITA", "transport": "IYT", "infra": "IGF", "software": "IGV", 
        "semis": "SMH", "cyber": "HACK", "media": "XLC", "biotech": "XBI", 
        "pharma": "XLV", "medtech": "IHI", "banks": "KBE", "brokers": "IAI", 
        "retail": "XRT", "staples": "XLP", "metals": "XME", "energy": "XLE", 
        "uranium": "URA", "chem": "XLB", "utilities": "XLU", "real_estate": "XLRE"
    }
}

@app.route("/api/equity-engine-v3")
def api_v3():
    try:
        start_str = request.args.get("start_date", "2015-01-01")
        x_win = int(request.args.get("opt_window_x", 36))
        y_pa = float(request.args.get("hurdle_y_pa", 0.02))
        z_lock = int(request.args.get("lock_z", 6))
        cost_rate = float(request.args.get("cost_rate", 0.001))

        mapping = TICKERS_V3["sectors"]
        all_t = [TICKERS_V3["equities"], TICKERS_V3["cash"]] + list(mapping.values())
        raw_data = yf.download(all_t + ["SHV", "XLK", "XLI", "XLF"], start="2000-01-01", auto_adjust=True, progress=False)["Close"]
        
        prices_raw = raw_data.resample("ME").last().ffill()
        # Padding Logik
        padding = [("XLC","XLK"), ("BIL","SHV"), ("ITA","XLI"), ("KBE","XLF")]
        for tk, ref in padding:
            if tk in prices_raw.columns and ref in prices_raw.columns:
                fv = prices_raw[tk].first_valid_index()
                if fv: prices_raw[tk] = prices_raw[tk].fillna(prices_raw[ref] * (prices_raw[tk].loc[fv] / prices_raw[ref].loc[fv]))

        inv_map = {v: k for k, v in mapping.items()}
        inv_map[TICKERS_V3["equities"]] = "equities"; inv_map[TICKERS_V3["cash"]] = "cash"
        prices = prices_raw.rename(columns=inv_map).dropna(subset=["equities", "cash"])

        start_i = np.where(prices.index >= pd.to_datetime(start_str))[0][0]
        if start_i < x_win: start_i = x_win

        engine_rets, acwi_rets, dates, weight_hist = [], [], [], []
        curr_p, months_active, prev_w, mode = None, 0, {}, "INIT"

        for i in range(start_i, len(prices)-1):
            # Optimierungs-Regel: Alle z Monate ODER jeden Monat wenn nicht in Alpha
            if curr_p is None or months_active >= z_lock or mode != "ALPHA":
                best = find_best_params(prices.iloc[i-x_win:i], list(mapping.keys()), y_pa/12)
                curr_p = best if best else {'w1': 0.0, 'w3': 0.0, 'w6': 0.2, 'w12': 0.8}
                months_active = 0
            
            # Signal Check
            acwi_s = compute_score(prices["equities"], i, **curr_p)
            bil_s = compute_score(prices["cash"], i, **curr_p)
            s_scores = {s: compute_score(prices[s], i, **curr_p) for s in mapping.keys()}
            qualified = {s: sc for s, sc in s_scores.items() if sc > (acwi_s + (y_pa/12)) and sc > bil_s}
            
            f_w = {}
            if qualified:
                top = sorted(qualified, key=qualified.get, reverse=True)[:5]
                for s in top: f_w[s] = 1.0/len(top)
                mode = "ALPHA"
            elif acwi_s > bil_s:
                f_w["equities"] = 1.0
                mode = "BETA"
            else:
                f_w["cash"] = 1.0
                mode = "CASH"

            turnover = sum(abs(f_w.get(a,0) - prev_w.get(a,0)) for a in set(f_w)|set(prev_w))
            m_ret = sum(w * (prices[a].iloc[i+1]/prices[a].iloc[i]-1) for a, w in f_w.items()) - (turnover * cost_rate)
            
            engine_rets.append(float(m_ret)); acwi_rets.append(float(prices["equities"].iloc[i+1]/prices["equities"].iloc[i]-1))
            dates.append(prices.index[i+1]); weight_hist.append({"date": prices.index[i+1].strftime("%Y-%m-%d"), "weights": f_w, "mode": mode})
            prev_w, months_active = f_w, months_active + 1

        return jsonify({
            "series": {"dates": [d.strftime("%Y-%m-%d") for d in dates], "equity_engine": (1 + pd.Series(engine_rets)).cumprod().tolist(), "acwi": (1 + pd.Series(acwi_rets)).cumprod().tolist()},
            "performance": {"equity_engine": calc_stats(pd.Series(engine_rets, index=dates)), "acwi": calc_stats(pd.Series(acwi_rets, index=dates))},
            "weight_history": weight_hist
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
