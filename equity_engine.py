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
        "cagr": cagr,
        "vola": vola,
        "max_drawdown": float((curr / curr.cummax() - 1).min()),
        "sharpe": sharpe,
        "total_return": float(curr.iloc[-1] - 1)
    }

def compute_score(series, i):
    """ Momentum-Score: 50% 1M, 30% 3M, 20% 6M """
    try:
        p = series
        return 0.5*(p.iloc[i]/p.iloc[i-1]-1) + 0.3*(p.iloc[i]/p.iloc[i-3]-1) + 0.2*(p.iloc[i]/p.iloc[i-6]-1)
    except:
        return 0

# --- Konfiguration ---

TICKERS = {
    "us_long_history": {
        "equities": "ACWI",
        "energy": "XLE", "materials": "XLB", "industrials": "XLI", "cons_disc": "XLY",
        "cons_staples": "XLP", "health_care": "XLV", "financials": "XLF", "technology": "XLK",
        "utilities": "XLU", "real_estate": "IYR", "semis": "SMH", "software": "IGV"
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
        s_set = request.args.get("sector_proxy_set", "us_long_history")
        cost_rate = float(request.args.get("cost_rate", 0.001))
        start_str = request.args.get("start_date", "2009-03-31")
        
        mapping = TICKERS.get(s_set, TICKERS["us_long_history"])
        tickers = list(mapping.values())
        
        # Daten laden
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
            eq_score = compute_score(prices["equities"], i)
            sectors = [c for c in prices.columns if c != "equities"]
            scores = {s: compute_score(prices[s], i) for s in sectors}
            
            # --- Strategie-Logik: Hürde & Selektion ---
            # Hürde leicht gesenkt auf 10% für bessere Reaktionszeit
            huerde = eq_score * 1.1 if eq_score > 0 else eq_score + 0.01
            strong_sectors = {s: sc for s, sc in scores.items() if sc > huerde}
            
            sector_weights = {}
            
            # Szenario: Sektoren identifizieren
            if strong_sectors:
                # STABILISIERT: Summe der Absolutwerte gegen Division durch Null
                total_s_score = sum(abs(sc) for sc in strong_sectors.values())
                if total_s_score > 1e-9:
                    for s, sc in strong_sectors.items():
                        sector_weights[s] = 0.70 * (abs(sc) / total_s_score)
                else:
                    for s in strong_sectors:
                        sector_weights[s] = 0.70 / len(strong_sectors)
            
            # --- DER VERBESSERTE RE-INVEST-FILTER (Minimiert Cash-Drag) ---
            # 30% Basis-ACWI ist das Minimum
            temp_weights = {"equities": 0.30}
            
            if not strong_sectors:
                # Wenn kein Sektor die Hürde nimmt -> Volle 70% zurück in ACWI
                temp_weights["equities"] += 0.70
            else:
                # Sektoren gewichten, aber bei negativem Momentum zurück in ACWI schieben
                for s, w in sector_weights.items():
                    if scores.get(s, 0) > 0:
                        temp_weights[s] = w
                    else:
                        temp_weights["equities"] += w

            # --- FINALE CASH-PRÜFUNG ---
            # Wir gehen nur in Cash, wenn der Markt (ACWI) selbst negatives Momentum hat
            final_weights = {}
            for asset, weight in temp_weights.items():
                asset_score = eq_score if asset == "equities" else scores.get(asset, 0)
                if asset_score > 0:
                    final_weights[asset] = weight
                # Negativer Score führt hier zu Cash (0% Rendite)

            # Performance-Berechnung inklusive Turnover-Kosten
            all_assets = set(final_weights.keys()) | set(prev_w.keys())
            turnover = sum(abs(final_weights.get(a, 0) - prev_w.get(a, 0)) for a in all_assets)
            
            m_ret = sum(final_weights[a] * rets[a].iloc[i+1] for a in final_weights if a in rets.columns)
            m_ret -= (turnover * cost_rate)
            
            engine_rets.append(float(m_ret))
            acwi_rets.append(float(rets["equities"].iloc[i+1]))
            dates.append(prices.index[i+1])
            
            weight_hist.append({
                "date": prices.index[i+1].strftime("%Y-%m-%d"), 
                "selected": [k for k, v in final_weights.items() if k != "equities"], 
                "weights": final_weights
            })
            prev_w = final_weights

        # DataFrames für Statistiken
        df_engine = pd.Series(engine_rets, index=dates)
        df_acwi = pd.Series(acwi_rets, index=dates)
        
        df_m = pd.DataFrame({"ret": engine_rets}, index=dates)
        matrix = df_m.groupby([df_m.index.year, df_m.index.month])["ret"].sum().unstack().replace([np.inf, -np.inf], 0).fillna(0)

        return jsonify({
            "series": {
                "dates": [d.strftime("%Y-%m-%d") for d in dates],
                "equity_engine": (1 + df_engine).cumprod().tolist(),
                "acwi": (1 + df_acwi).cumprod().tolist()
            },
            "matrix": matrix.to_dict(orient="index"),
            "performance": {
                "equity_engine": calc_stats(df_engine),
                "acwi": calc_stats(df_acwi)
            },
            "latest_weights": weight_hist[-1] if weight_hist else {},
            "weight_history": weight_hist
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
