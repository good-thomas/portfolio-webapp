import math
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# --- Hilfsfunktionen ---

def calc_stats(r):
    if r.empty:
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
        p = series
        # Momentum Berechnung auf Basis der Monats-Schlusskurse
        ret1 = (p.iloc[i]/p.iloc[i-1]-1)
        ret3 = (p.iloc[i]/p.iloc[i-3]-1)
        ret6 = (p.iloc[i]/p.iloc[i-6]-1)
        ret12 = (p.iloc[i]/p.iloc[i-12]-1)
        return (w1 * ret1) + (w3 * ret3) + (w6 * ret6) + (w12 * ret12)
    except:
        return -999 # Signal für unzureichende Daten

# --- V3 OPTIMIERUNGS-KERN ---

def find_best_params(price_slice, sector_list, y_monthly_premium):
    # Reduzierte Test-Kombinationen für Speed
    test_configs = [
        {'w1': 0.0, 'w3': 0.0, 'w6': 0.0, 'w12': 1.0},
        {'w1': 0.0, 'w3': 0.0, 'w6': 0.5, 'w12': 0.5},
        {'w1': 0.0, 'w3': 0.2, 'w6': 0.4, 'w12': 0.4},
        {'w1': 0.2, 'w3': 0.2, 'w6': 0.2, 'w12': 0.4}
    ]
    
    best_cfg = None
    max_sharpe = -1
    
    acwi_rets = price_slice["equities"].pct_change().dropna()
    acwi_stats = calc_stats(acwi_rets)
    target_cagr = acwi_stats["cagr"] + (y_monthly_premium * 12)

    # Simulation nur an jedem 3. Monatspunkt im Fenster spart massiv Zeit
    step_range = range(12, len(price_slice)-1, 3)

    for cfg in test_configs:
        sim_rets = []
        for j in step_range:
            bil_s = compute_score(price_slice["cash"], j, **cfg)
            acwi_s = compute_score(price_slice["equities"], j, **cfg)
            
            # Sektor-Wahl
            scores = {s: compute_score(price_slice[s], j, **cfg) for s in sector_list}
            qualified = [s for s, sc in scores.items() if sc > (acwi_s + y_monthly_premium) and sc > bil_s]
            
            if qualified:
                m_ret = price_slice[qualified[:3]].pct_change().iloc[j+1].mean()
            elif acwi_s > bil_s:
                m_ret = acwi_rets.iloc[j]
            else:
                m_ret = price_slice["cash"].pct_change().iloc[j+1]
            sim_rets.append(m_ret)
        
        res = calc_stats(pd.Series(sim_rets))
        if res["cagr"] > target_cagr and res["sharpe"] > acwi_stats["sharpe"]:
            if res["sharpe"] > max_sharpe:
                max_sharpe = res["sharpe"]
                best_cfg = cfg
                
    return best_cfg
# --- KONFIGURATION ---

TICKERS = {
    "us_long_history": {
        "equities": "ACWI",
        "cash": "BIL",
        "defense": "ITA", "transport": "IYT", "infra_build": "IGF",
        "software": "IGV", "semis": "SMH", "cybersecurity": "HACK", "media_comm": "XLC",
        "biotech": "XBI", "healthcare_pharma": "XLV", "medtech": "IHI",
        "banks": "KBE", "insurance_brokers": "IAI",
        "retail": "XRT", "staples": "XLP",
        "metals_mining": "XME", "energy_oil_gas": "XLE", "uranium": "URA", "materials_chem": "XLB",
        "utilities": "XLU", "real_estate": "XLRE"
    }
}

@app.route("/api/equity-engine-v3")
def api_v3():
    try:
        # Parameter aus Request
        x_window = int(request.args.get("opt_window_x", 36))
        y_premium_pa = float(request.args.get("hurdle_y_pa", 0.02))
        y_monthly = y_premium_pa / 12
        z_lock = int(request.args.get("lock_z", 6))
        cost_rate = float(request.args.get("cost_rate", 0.001))
        start_str = request.args.get("start_date", "2011-01-01")

        mapping = TICKERS["us_long_history"]
        all_tickers = list(mapping.values())
        
        # Proxies für das Padding
        proxies = ["XLI", "XLK", "XLF", "XLY", "SHV"]
        raw = yf.download(all_tickers + proxies, start="2000-01-01", auto_adjust=True, progress=False)["Close"]
        prices_raw = raw.resample("ME").last().ffill()

        # Ratio-Padding
        padding_pairs = [
            ("XLC", "XLK"), ("XLRE", "ACWI"), ("SMH", "XLK"), ("HACK", "IGV"), ("URA", "XLE"), 
            ("ITA", "XLI"), ("IYT", "XLI"), ("IGV", "XLK"), ("KBE", "XLF"), ("XME", "XLB"), 
            ("XRT", "XLY"), ("XBI", "XLV"), ("IAI", "XLF"), ("IHI", "XLV"), ("IGF", "XLI"), ("BIL", "SHV")
        ]
        for tk, ref in padding_pairs:
            if tk in prices_raw.columns and ref in prices_raw.columns and prices_raw[tk].isnull().any():
                fv = prices_raw[tk].first_valid_index()
                if fv:
                    ratio = prices_raw[tk].loc[fv] / prices_raw[ref].loc[fv]
                    prices_raw[tk] = prices_raw[tk].fillna(prices_raw[ref] * ratio)

        inv_map = {v: k for k, v in mapping.items()}
        prices = prices_raw[all_tickers].rename(columns=inv_map).dropna()
        rets = prices.pct_change()

        # Start-Punkt finden (mindestens x_window Daten für erste Optimierung)
        requested_start = pd.to_datetime(start_str)
        start_idx = np.where(prices.index >= requested_start)[0][0]
        if start_idx < x_window: start_idx = x_window

        # Backtest Variablen
        engine_rets, acwi_rets, dates, weight_hist, param_log = [], [], [], [], []
        curr_params = None
        months_since_opt = 0
        prev_w = {}
        sector_list = [c for c in prices.columns if c not in ["equities", "cash"]]

        # --- HAUPTSCHLEIFE ---
        for i in range(start_idx, len(prices)-1):
            # 1. OPTIMIERUNGS-LOGIK
            do_optimize = False
            if curr_params is None: do_optimize = True
            elif curr_params == "ACWI_FALLBACK": do_optimize = True # Jeden Monat im Fallback prüfen
            elif months_since_opt >= z_lock: do_optimize = True
            
            if do_optimize:
                window_data = prices.iloc[i-x_window:i]
                best = find_best_params(window_data, sector_list, y_monthly)
                curr_params = best if best else "ACWI_FALLBACK"
                months_since_opt = 0
            
            # 2. SIGNAL-BERECHNUNG (Double-Gate)
            if curr_params == "ACWI_FALLBACK":
                # Standard-Parameter für Fallback-Check nutzen (z.B. 12M)
                p = {"w1": 0, "w3": 0, "w6": 0, "w12": 1.0}
                mode_label = "INDEX_FALLBACK"
            else:
                p = curr_params
                mode_label = f"ALPHA_{p['w12']}M_{p['w6']}M"

            bil_s = compute_score(prices["cash"], i, **p)
            acwi_s = compute_score(prices["equities"], i, **p)
            sector_scores = {s: compute_score(prices[s], i, **p) for s in sector_list}
            
            # Thomas-Regel: Sektor > ACWI+y UND Sektor > BIL
            qualified = {s: sc for s, sc in sector_scores.items() if sc > (acwi_s + y_monthly) and sc > bil_s}
            
            final_weights = {}
            if qualified:
                top_s = sorted(qualified, key=qualified.get, reverse=True)[:5]
                for s in top_s: final_weights[s] = 1.0 / len(top_s)
            elif acwi_s > bil_s:
                final_weights["equities"] = 1.0
            else:
                final_weights["cash"] = 1.0 # Crash-Schutz

            # 3. PERFORMANCE & LOGGING
            turnover = sum(abs(final_weights.get(a, 0) - prev_w.get(a, 0)) for a in set(final_weights) | set(prev_w))
            m_ret = sum(w * rets[a].iloc[i+1] for a, w in final_weights.items()) - (turnover * cost_rate)
            
            engine_rets.append(float(m_ret))
            acwi_rets.append(float(rets["equities"].iloc[i+1]))
            dates.append(prices.index[i+1])
            weight_hist.append({"date": prices.index[i+1].strftime("%Y-%m-%d"), "weights": final_weights, "mode": mode_label})
            prev_w = final_weights
            months_since_opt += 1

        df_engine = pd.Series(engine_rets, index=dates)
        df_acwi = pd.Series(acwi_rets, index=dates)

        return jsonify({
            "series": {
                "dates": [d.strftime("%Y-%m-%d") for d in dates],
                "equity_engine": (1 + df_engine).cumprod().tolist(),
                "acwi": (1 + df_acwi).cumprod().tolist()
            },
            "performance": {
                "equity_engine": calc_stats(df_engine),
                "acwi": calc_stats(df_acwi)
            },
            "weight_history": weight_hist
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
