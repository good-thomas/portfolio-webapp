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
    sharpe = float(cagr / vola) if vola > 0 else 0
    return {
        "cagr": round(cagr, 4), "vola": round(vola, 4), 
        "max_drawdown": round(float((curr / curr.cummax() - 1).min()), 4),
        "sharpe": round(sharpe, 4), "total_return": round(float(curr.iloc[-1] - 1), 4)
    }

def compute_score(series, i, w1, w3, w6, w12):
    try:
        if i < 12 or i >= len(series): return -999
        p = series
        # Absicherung gegen NaN Werte
        if pd.isna(p.iloc[i]) or pd.isna(p.iloc[i-12]): return -999
        
        ret1 = (p.iloc[i]/p.iloc[i-1]-1)
        ret3 = (p.iloc[i]/p.iloc[i-3]-1)
        ret6 = (p.iloc[i]/p.iloc[i-6]-1)
        ret12 = (p.iloc[i]/p.iloc[i-12]-1)
        return (w1 * ret1) + (w3 * ret3) + (w6 * ret6) + (w12 * ret12)
    except:
        return -999

# --- V3 Kern: Optimierung ---

def find_best_params(price_slice, sector_list, y_monthly_premium):
    # Nur 4 strategische Archetypen für maximalen Speed
    test_configs = [
        {'w1': 0.0, 'w3': 0.0, 'w6': 0.0, 'w12': 1.0}, # Momentum 12M
        {'w1': 0.0, 'w3': 0.0, 'w6': 0.5, 'w12': 0.5}, # 6M/12M Mix
        {'w1': 0.0, 'w3': 0.3, 'w6': 0.4, 'w12': 0.3}, # Mittelfristig
        {'w1': 0.2, 'w3': 0.3, 'w6': 0.3, 'w12': 0.2}  # Reaktiv
    ]
    
    best_cfg = test_configs[0]
    max_sharpe = -1
    
    # Benchmark ACWI
    acwi_rets = price_slice["equities"].pct_change().dropna()
    acwi_stats = calc_stats(acwi_rets)
    target_cagr = acwi_stats["cagr"] + (y_monthly_premium * 12)

    # Wir prüfen nur jeden 3. Monat im Fenster x für Speed
    sim_indices = range(12, len(price_slice)-1, 3)

    for cfg in test_configs:
        sim_rets = []
        for j in sim_indices:
            bil_s = compute_score(price_slice["cash"], j, **cfg)
            acwi_s = compute_score(price_slice["equities"], j, **cfg)
            
            sector_scores = {s: compute_score(price_slice[s], j, **cfg) for s in sector_list}
            # Thomas-Regel: Sektor > ACWI+y UND Sektor > BIL
            qualified = [s for s, sc in sector_scores.items() if sc > (acwi_s + y_monthly_premium) and sc > bil_s]
            
            if qualified:
                m_ret = price_slice[qualified[:3]].pct_change().iloc[j+1].mean()
            elif acwi_s > bil_s:
                m_ret = price_slice["equities"].pct_change().iloc[j+1]
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

# --- Konfiguration & API ---

TICKERS_V3 = {
    "equities": "ACWI",
    "cash": "BIL",
    "sectors": {
        "defense": "ITA", "transport": "IYT", "infra": "IGF",
        "software": "IGV", "semis": "SMH", "cyber": "HACK", "media": "XLC",
        "biotech": "XBI", "pharma": "XLV", "medtech": "IHI",
        "banks": "KBE", "brokers": "IAI", "retail": "XRT", "staples": "XLP",
        "metals": "XME", "energy": "XLE", "uranium": "URA", "chem": "XLB",
        "utilities": "XLU", "real_estate": "XLRE"
    }
}

@app.route("/api/equity-engine-v3")
def api_v3():
    try:
        print("Starte Engine v3...")
        # 1. Parameter laden
        start_str = request.args.get("start_date", "2015-01-01")
        x_window = int(request.args.get("opt_window_x", 36))
        y_premium_pa = float(request.args.get("hurdle_y_pa", 0.02))
        z_lock = int(request.args.get("lock_z", 6))
        cost_rate = float(request.args.get("cost_rate", 0.001))
        
        # 2. Daten abrufen
        mapping = TICKERS_V3["sectors"]
        all_tickers = [TICKERS_V3["equities"], TICKERS_V3["cash"]] + list(mapping.values())
        proxies = ["XLI", "XLK", "XLF", "XLY", "SHV"]
        
        print("Lade Yahoo Daten...")
        raw = yf.download(all_tickers + proxies, start="2000-01-01", auto_adjust=True, progress=False)["Close"]
        if raw.empty: raise ValueError("Keine Daten von Yahoo Finance erhalten.")
        
        prices_raw = raw.resample("ME").last().ffill()

        # Ratio-Padding (v2 Logik)
        padding = [("XLC","XLK"), ("XLRE","ACWI"), ("SMH","XLK"), ("HACK","IGV"), ("URA","XLE"), ("BIL","SHV"), ("ITA","XLI")]
        for tk, ref in padding:
            if tk in prices_raw.columns and ref in prices_raw.columns:
                fv = prices_raw[tk].first_valid_index()
                if fv:
                    ratio = prices_raw[tk].loc[fv] / prices_raw[ref].loc[fv]
                    prices_raw[tk] = prices_raw[tk].fillna(prices_raw[ref] * ratio)

        # Mapping auf interne Namen
        inv_map = {v: k for k, v in mapping.items()}
        inv_map[TICKERS_V3["equities"]] = "equities"
        inv_map[TICKERS_V3["cash"]] = "cash"
        
        prices = prices_raw[all_tickers].rename(columns=inv_map)
        # Wir droppen nur Zeilen, die GAR KEINE Daten haben, um Lücken in einzelnen ETFs zu erlauben
        prices = prices.dropna(how='all')
        rets = prices.pct_change()

        # 3. Startindex finden
        req_start = pd.to_datetime(start_str)
        start_i = np.where(prices.index >= req_start)[0][0]
        if start_i < x_window: start_i = x_window
        
        engine_rets, acwi_rets, dates, weight_hist = [], [], [], []
        curr_params, months_active, prev_w = None, 0, {}
        sector_list = list(mapping.keys())

        # 4. Backtest-Loop
        print(f"Starte Loop ab {prices.index[start_i]}...")
        for i in range(start_i, len(prices)-1):
            # Optimierung?
            if curr_params is None or months_active >= z_lock or curr_params == "INDEX_ONLY":
                window_data = prices.iloc[i-x_window:i]
                best = find_best_params(window_data, sector_list, y_premium_pa/12)
                curr_params = best if best else "INDEX_ONLY"
                months_active = 0
            
            # Signal-Check
            p = curr_params if isinstance(curr_params, dict) else {'w1':0, 'w3':0, 'w6':0, 'w12':1}
            mode = "ALPHA" if isinstance(curr_params, dict) else "BETA"
            
            bil_s = compute_score(prices["cash"], i, **p)
            acwi_s = compute_score(prices["equities"], i, **p)
            s_scores = {s: compute_score(prices[s], i, **p) for s in sector_list}
            
            qualified = {s: sc for s, sc in s_scores.items() if sc > (acwi_s + (y_premium_pa/12)) and sc > bil_s}
            
            f_weights = {}
            if qualified:
                top = sorted(qualified, key=qualified.get, reverse=True)[:5]
                for s in top: f_weights[s] = 1.0/len(top)
            elif acwi_s > bil_s:
                f_weights["equities"] = 1.0
                mode = "BETA"
            else:
                f_weights["cash"] = 1.0
                mode = "CASH"

            # Performance
            turnover = sum(abs(f_weights.get(a,0) - prev_w.get(a,0)) for a in set(f_weights)|set(prev_w))
            m_ret = sum(w * rets[a].iloc[i+1] for a, w in f_weights.items()) - (turnover * cost_rate)
            
            engine_rets.append(float(m_ret))
            acwi_rets.append(float(rets["equities"].iloc[i+1]))
            dates.append(prices.index[i+1])
            weight_hist.append({"date": prices.index[i+1].strftime("%Y-%m-%d"), "weights": f_weights, "mode": mode})
            
            prev_w, months_active = f_weights, months_active + 1

        print("Berechnung fertig.")
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
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
