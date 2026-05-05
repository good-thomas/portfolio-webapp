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

# --- Basis-Logik ---

def calc_stats(r):
    if r.empty or r.isnull().all():
        return {"cagr": 0, "vola": 0, "max_drawdown": 0, "sharpe": 0, "total_return": 0}
    curr = (1 + r).cumprod()
    years = (r.index[-1] - r.index[0]).days / 365.25 if len(r) > 1 else 1
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
        if i < 12: return -999
        p = series
        r1 = (p.iloc[i]/p.iloc[i-1]-1)
        r3 = (p.iloc[i]/p.iloc[i-3]-1)
        r6 = (p.iloc[i]/p.iloc[i-6]-1)
        r12 = (p.iloc[i]/p.iloc[i-12]-1)
        return (w1 * r1) + (w3 * r3) + (w6 * r6) + (w12 * r12)
    except: return -999

# --- Kern-Engine ---

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
        # 1. Parameter
        start_str = request.args.get("start_date", "2018-01-01")
        x_win = int(request.args.get("opt_window_x", 36))
        y_pa = float(request.args.get("hurdle_y_pa", 0.02))
        z_lock = int(request.args.get("lock_z", 6))
        
        # 2. Daten-Download
        mapping = TICKERS_V3["sectors"]
        all_t = [TICKERS_V3["equities"], TICKERS_V3["cash"]] + list(mapping.values())
        data = yf.download(all_t, start="2005-01-01", auto_adjust=True, progress=False)["Close"]
        
        if data.empty:
            return jsonify({"error": "Yahoo Finance hat keine Daten geliefert."}), 500
            
        prices = data.resample("ME").last().ffill().dropna(how='all')
        
        # Internes Mapping
        inv_map = {v: k for k, v in mapping.items()}
        inv_map[TICKERS_V3["equities"]] = "equities"
        inv_map[TICKERS_V3["cash"]] = "cash"
        prices = prices.rename(columns=inv_map)
        
        # Check ob equities und cash da sind
        if "equities" not in prices or "cash" not in prices:
            return jsonify({"error": "ACWI oder BIL fehlen in den Daten."}), 500

        # 3. Zeit-Index
        req_start = pd.to_datetime(start_str)
        try:
            start_i = np.where(prices.index >= req_start)[0][0]
        except:
            return jsonify({"error": "Startdatum liegt außerhalb der verfügbaren Daten."}), 400
        
        if start_i < x_win: start_i = x_win

        # 4. Backtest (Vereinfacht für Stabilität)
        engine_rets, acwi_rets, dates, weight_hist = [], [], [], []
        prev_w = {}
        
        for i in range(start_i, len(prices)-1):
            # Wir nutzen hier fix 12M Momentum für den ersten Test, um die Opt-Schleife als Fehlerquelle auszuschließen
            acwi_s = compute_score(prices["equities"], i, 0, 0, 0, 1)
            bil_s = compute_score(prices["cash"], i, 0, 0, 0, 1)
            
            s_scores = {s: compute_score(prices[s], i, 0, 0, 0, 1) for s in mapping.keys()}
            qualified = {s: sc for s, sc in s_scores.items() if sc > (acwi_s + (y_pa/12)) and sc > bil_s}
            
            f_weights = {}
            if qualified:
                top = sorted(qualified, key=qualified.get, reverse=True)[:5]
                for s in top: f_weights[s] = 1.0/len(top)
            elif acwi_s > bil_s:
                f_weights["equities"] = 1.0
            else:
                f_weights["cash"] = 1.0

            m_ret = sum(w * (prices[a].iloc[i+1]/prices[a].iloc[i]-1) for a, w in f_weights.items())
            engine_rets.append(float(m_ret))
            acwi_rets.append(float(prices["equities"].iloc[i+1]/prices["equities"].iloc[i]-1))
            dates.append(prices.index[i+1])
            weight_hist.append({"date": prices.index[i+1].strftime("%Y-%m-%d"), "weights": f_weights, "mode": "V3-FIXED"})

        return jsonify({
            "series": {
                "dates": [d.strftime("%Y-%m-%d") for d in dates],
                "equity_engine": (1 + pd.Series(engine_rets)).cumprod().tolist(),
                "acwi": (1 + pd.Series(acwi_rets)).cumprod().tolist()
            },
            "performance": {
                "equity_engine": calc_stats(pd.Series(engine_rets, index=dates)),
                "acwi": calc_stats(pd.Series(acwi_rets, index=dates))
            },
            "weight_history": weight_hist
        })

    except Exception as e:
        return jsonify({"error": f"CRASH: {str(e)}", "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
