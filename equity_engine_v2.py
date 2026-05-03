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
        "cagr": cagr, "vola": vola, 
        "max_drawdown": float((curr / curr.cummax() - 1).min()),
        "sharpe": sharpe, "total_return": float(curr.iloc[-1] - 1)
    }

def compute_score(series, i, w1, w3, w6):
    """ Dynamisches Scoring-Modell basierend auf variablen Gewichten """
    try:
        p = series
        ret1 = (p.iloc[i]/p.iloc[i-1]-1)
        ret3 = (p.iloc[i]/p.iloc[i-3]-1)
        ret6 = (p.iloc[i]/p.iloc[i-6]-1)
        return (w1 * ret1) + (w3 * ret3) + (w6 * ret6)
    except:
        return 0

# --- Konfiguration ---

# --- Neue, agnostische Ticker-Struktur v3 ---
TICKERS = {
    "us_long_history": {  # Wir behalten den Namen bei, damit das Frontend nicht bricht
        "equities": "ACWI",
        # IT & Communication
        "software": "IGV", "semis": "SMH", "tech": "XLK", "media": "XLC",
        # Industrials
        "defense": "ITA", "transport": "IYT", "industrials": "XLI",
        # Health Care
        "biotech": "XBI", "healthcare": "XLV",
        # Financials
        "banks": "KBE", "insurance": "KIE",
        # Consumer
        "retail": "XRT", "staples": "XLP",
        # Resources & Infrastructure
        "metals": "XME", "materials": "XLB", "energy": "XLE", 
        "utilities": "XLU", "real_estate": "XLRE"
    },
    "ucits": {
        "equities": "ACWI",
        "energy": "WNRG.DE", "materials": "XMWS.DE", "technology": "IGPT.DE", "banks": "EXV1.DE",
        "health": "WHEA.DE", "utilities": "WUTI.DE", "real_estate": "DPRE.DE"
    }
}

@app.route("/api/equity-engine")
def api():
    try:
        # --- VARIABLE PARAMETER (VOM FRONTEND GESTEUERT) ---
        s_set = request.args.get("sector_proxy_set", "us_long_history")
        cost_rate = float(request.args.get("cost_rate", 0.001))
        start_str = request.args.get("start_date", "2009-03-31")
        
        # Scoring-Gewichte
        w1 = float(request.args.get("lookback_1m", 0.5))
        w3 = float(request.args.get("lookback_3m", 0.3))
        w6 = float(request.args.get("lookback_6m", 0.2))
        
        # Selektions-Parameter
        huerde_factor = float(request.args.get("selection_huerde", 1.1)) # Multiplikator für ACWI-Score
        max_sectors = int(request.args.get("max_sectors", 3))
        sector_limit = float(request.args.get("sector_weight_total", 0.70))

        # --- DATEN-PROZESSING ---
        mapping = TICKERS.get(s_set, TICKERS["us_long_history"])
        tickers = list(mapping.values())
        raw = yf.download(tickers, start="2000-01-01", auto_adjust=True, progress=False)["Close"]
        inv_map = {v: k for k, v in mapping.items()}
        prices = raw.rename(columns=inv_map).resample("ME").last().ffill().dropna()
        rets = prices.pct_change()

        try:
            start_i = np.where(prices.index >= pd.to_datetime(start_str))[0][0]
        except:
            start_i = 12

        engine_rets, acwi_rets, dates, weight_hist = [], [], [], []
        prev_w = {}

        for i in range(start_i, len(prices)-1):
            eq_score = compute_score(prices["equities"], i, w1, w3, w6)
            sectors = [c for c in prices.columns if c != "equities"]
            scores = {s: compute_score(prices[s], i, w1, w3, w6) for s in sectors}
            
            # --- OPPORTUNISTISCHE SELEKTION ---
            huerde = eq_score * huerde_factor if eq_score > 0 else eq_score + 0.01
            # Filter: Nur Sektoren über Hürde UND mit positivem Momentum
            qualified = {s: sc for s, sc in scores.items() if sc > huerde and sc > 0}
            
            # Konzentration: Nimm nur die Top X Sektoren
            top_sectors = sorted(qualified, key=qualified.get, reverse=True)[:max_sectors]
            
            sector_weights = {}
            base_acwi_weight = 1.0 - sector_limit # Der Teil, der immer im Markt bleibt
            
            if top_sectors:
                total_s_score = sum(abs(qualified[s]) for s in top_sectors)
                if total_s_score > 1e-9:
                    for s in top_sectors:
                        sector_weights[s] = sector_limit * (qualified[s] / total_s_score)
                else:
                    for s in top_sectors:
                        sector_weights[s] = sector_limit / len(top_sectors)
            
            # --- ZUSAMMENFÜHRUNG ---
            # Wir starten mit dem Basis-ACWI Anteil
            temp_weights = {"equities": base_acwi_weight}
            
            # Falls Sektor-Gewichte nicht voll ausgeschöpft (weil keine gefunden), 
            # fließt der Rest zurück in den ACWI (Agnostisches Prinzip)
            current_sector_total = sum(sector_weights.values())
            temp_weights["equities"] += (sector_limit - current_sector_total)
            
            for s, w in sector_weights.items():
                temp_weights[s] = w

            # --- RISK-OFF FILTER (CASH-LOGIK) ---
            final_weights = {}
            for asset, weight in temp_weights.items():
                asset_score = eq_score if asset == "equities" else scores.get(asset, 0)
                if asset_score > 0: # Nur investieren, wenn Momentum absolut positiv
                    final_weights[asset] = weight

            # Performance & Kosten
            all_assets = set(final_weights.keys()) | set(prev_w.keys())
            turnover = sum(abs(final_weights.get(a, 0) - prev_w.get(a, 0)) for a in all_assets)
            m_ret = sum(final_weights[a] * rets[a].iloc[i+1] for a in final_weights if a in rets.columns)
            m_ret -= (turnover * cost_rate)
            
            engine_rets.append(float(m_ret))
            acwi_rets.append(float(rets["equities"].iloc[i+1]))
            dates.append(prices.index[i+1])
            weight_hist.append({"date": prices.index[i+1].strftime("%Y-%m-%d"), "weights": final_weights})
            prev_w = final_weights

        # Resultate
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
