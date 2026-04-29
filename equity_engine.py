import math
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Konfiguration
DEFAULT_START = "2009-01-01"
TICKERS = {
    "us_long_history": {
        "equities": "ACWI",
        "energy": "XLE", "materials": "XLB", "industrials": "XLI", "cons_disc": "XLY",
        "cons_staples": "XLP", "health_care": "XLV", "financials": "XLF", "technology": "XLK",
        "utilities": "XLU", "real_estate": "IYR", "semis": "SMH", "software": "IGV"
    },
    "ucits": {
        "equities": "ACWI",
        "energy": "WNRG.DE", "materials": "XMWS.DE", "technology": "IGPT.DE", "banks": "EXV1.DE"
    }
}

def compute_score(series, i):
    try:
        p = series
        return 0.5*(p.iloc[i]/p.iloc[i-3]-1) + 0.3*(p.iloc[i]/p.iloc[i-6]-1) + 0.2*(p.iloc[i]/p.iloc[i-12]-1)
    except: return 0

@app.route("/api/equity-engine")
def api():
    try:
        # Parameter
        s_set = request.args.get("sector_proxy_set", "us_long_history")
        cost_rate = float(request.args.get("cost_rate", 0.001))
        start_str = request.args.get("start_date", DEFAULT_START)
        
        # Daten laden
        mapping = TICKERS.get(s_set, TICKERS["us_long_history"])
        tickers = list(mapping.values())
        raw = yf.download(tickers, start="2000-01-01", auto_adjust=True, progress=False)["Close"]
        
        # Mapping Ticker -> Name
        inv_map = {v: k for k, v in mapping.items()}
        prices = raw.rename(columns=inv_map).resample("ME").last().ffill().dropna()
        rets = prices.pct_change()

        # Backtest
        start_i = np.where(prices.index >= pd.to_datetime(start_str))[0][0]
        engine_rets, acwi_rets, dates, weight_hist = [], [], [], []
        prev_w = {}

        for i in range(start_i, len(prices)-1):
            eq_score = compute_score(prices["equities"], i)
            sectors = [c for c in prices.columns if c != "equities"]
            scores = {s: compute_score(prices[s], i) for s in sectors}
            
            top = sorted([s for s in scores if scores[s] > eq_score], key=lambda x: scores[x], reverse=True)[:3]
            
            w = {"equities": 0.5, **{s: (0.5/len(top)) for s in top}} if top else {"equities": 1.0}
            
            all_a = set(w.keys()) | set(prev_w.keys())
            to = sum(abs(w.get(a,0) - prev_w.get(a,0)) for a in all_a)
            
            m_ret = sum(w[a] * rets[a].iloc[i+1] for a in w) - (to * cost_rate)
            
            engine_rets.append(m_ret)
            acwi_rets.append(rets["equities"].iloc[i+1])
            dates.append(prices.index[i+1])
            weight_hist.append({"date": prices.index[i+1].strftime("%Y-%m-%d"), "selected": top, "weights": w})
            prev_w = w

        # Performance & Matrix
        df = pd.DataFrame({"ret": engine_rets}, index=dates)
        # Fix: value_counts() statt value_index()
        counts = pd.Series([s for h in weight_hist for s in h["selected"]]).value_counts().to_dict()
        matrix = df.groupby([df.index.year, df.index.month])["ret"].sum().unstack().replace([np.inf, -np.inf], 0).fillna(0)

        # Matrix-Bereinigung (NaN zu None für JSON-Kompatibilität)
        matrix_dict = matrix.where(pd.notnull(matrix), None).to_dict(orient="index")

        return jsonify({
            "series": {
                "dates": [d.strftime("%Y-%m-%d") for d in dates],
                "equity_engine": (1 + pd.Series(engine_rets)).cumprod().tolist(),
                "acwi": (1 + pd.Series(acwi_rets)).cumprod().tolist()
            },
            "matrix": matrix_dict,
            "selection_counts": counts,
            "performance": {
                "equity_engine": calc_stats(pd.Series(engine_rets, index=dates)),
                "acwi": calc_stats(pd.Series(acwi_rets, index=dates))
            },
            "latest_weights": weight_hist[-1] if weight_hist else {},
            "weight_history": weight_hist  # <--- DIESE ZEILE HAT GEFEHLT
        })
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
