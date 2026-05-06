import math, numpy as np, pandas as pd, yfinance as yf
from flask import Flask, request, jsonify
from flask_cors import CORS
import os, traceback

app = Flask(__name__)
CORS(app)

# --- MASTER ENGINE V3.5 (Thomas Edition: V3 Logic + Score Weighting) ---

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
        # 1. Parameter & Daten-Setup
        start_date = pd.to_datetime(request.args.get("start_date", "2015-01-01"))
        x_win = int(request.args.get("opt_window_x", 36))
        y_pa = float(request.args.get("hurdle_y_pa", 0.02)) / 12
        z_lock = int(request.args.get("lock_z", 6))
        
        mapping = TICKERS_V3["sectors"]
        all_t = [TICKERS_V3["equities"], TICKERS_V3["cash"]] + list(mapping.values())
        
        raw = yf.download(all_t + ["SHV", "XLK", "XLI"], start="2000-01-01", auto_adjust=True, progress=False)["Close"]
        df = raw.resample("ME").last().ffill()
        
        if "BIL" in df and "SHV" in df: df["BIL"] = df["BIL"].fillna(df["SHV"] * (df["BIL"].dropna().iloc[0]/df["SHV"].loc[df["BIL"].dropna().index[0]]))
        if "XLC" in df and "XLK" in df: df["XLC"] = df["XLC"].fillna(df["XLK"] * (df["XLC"].dropna().iloc[0]/df["XLK"].loc[df["XLK"].dropna().index[0]]))
        
        inv_map = {v: k for k, v in mapping.items()}
        inv_map[TICKERS_V3["equities"]] = "equities"; inv_map[TICKERS_V3["cash"]] = "cash"
        prices = df.rename(columns=inv_map)[[c for c in inv_map.values() if c in df.rename(columns=inv_map).columns]]
        rets_df = prices.pct_change().fillna(0)
        
        configs = [
            {'w6': 0.2, 'w12': 0.8}, {'w6': 0.5, 'w12': 0.5},
            {'w3': 0.3, 'w6': 0.4, 'w12': 0.3}, {'w1': 0.2, 'w3': 0.3, 'w6': 0.3, 'w12': 0.2}
        ]

        # 3. Backtest-Loop
        start_i = np.where(prices.index >= start_date)[0][0]
        if start_i < x_win: start_i = x_win
        
        res_rets, acwi_rets, dates, history = [], [], [], []
        curr_p, mode, months_active = configs[0], "INIT", 0
        sector_cols = list(mapping.keys())

        for i in range(start_i, len(prices)-1):
            # Optimierung der Regel (V3-Stil: Erster Treffer)
            if months_active >= z_lock or mode != "ALPHA":
                best_cfg = configs[0]
                window_rets = rets_df.iloc[i-x_win:i]
                
                for cfg in configs:
                    s_score = (window_rets * cfg.get('w1', 0) + window_rets.rolling(3).mean() * cfg.get('w3', 0) + 
                               window_rets.rolling(6).mean() * cfg.get('w6', 0) + window_rets.rolling(12).mean() * cfg.get('w12', 0)).iloc[-1]
                    if s_score.max() > 0: best_cfg = cfg; break 
                curr_p = best_cfg
                months_active = 0

            # Signal Check
            def get_s(col):
                p_vec = prices[col]
                return (curr_p.get('w1',0)*(p_vec.iloc[i]/p_vec.iloc[i-1]-1) + 
                        curr_p.get('w6',0)*(p_vec.iloc[i]/p_vec.iloc[i-6]-1) + 
                        curr_p.get('w12',0)*(p_vec.iloc[i]/p_vec.iloc[i-12]-1))

            scores = {s: get_s(s) for s in sector_cols}
            eq_s, cash_s = get_s("equities"), get_s("cash")
            
            # Identifizierung der qualifizierten Sektoren
            qual = {s: sc for s, sc in scores.items() if sc > (eq_s + y_pa) and sc > cash_s}
            
            weights = {}
            if qual:
                top_keys = sorted(qual, key=qual.get, reverse=True)[:5]
                # NEU: Score-proportionale Gewichtung statt 1.0/len(top)
                sum_scores = sum(qual[k] for k in top_keys)
                for k in top_keys:
                    weights[k] = round(qual[k] / sum_scores, 4)
                mode = "ALPHA"
            elif eq_s > cash_s:
                weights["equities"] = 1.0; mode = "BETA"
            else:
                weights["cash"] = 1.0; mode = "CASH"

            m_ret = sum(w * rets_df[a].iloc[i+1] for a, w in weights.items())
            res_rets.append(m_ret); acwi_rets.append(rets_df["equities"].iloc[i+1])
            dates.append(prices.index[i+1])
            history.append({"date": prices.index[i+1].strftime("%Y-%m-%d"), "weights": weights, "mode": mode})
            months_active += 1

        def get_p_stats(r_list):
            r_ser = pd.Series(r_list)
            if r_ser.empty: return {"cagr": 0, "max_drawdown": 0, "sharpe": 0}
            cum = (1 + r_ser).cumprod()
            y = len(r_ser) / 12
            cagr = float(cum.iloc[-1]**(1/y) - 1) if y > 0 else 0
            dd = float((cum / cum.cummax() - 1).min())
            vola = float(r_ser.std() * math.sqrt(12))
            return {"cagr": round(cagr, 4), "max_drawdown": round(dd, 4), "sharpe": round(cagr / vola, 4) if vola > 0 else 0}

        return jsonify({
            "series": {
                "dates": [d.strftime("%Y-%m-%d") for d in dates], 
                "equity_engine": (1 + pd.Series(res_rets)).cumprod().tolist(), 
                "acwi": (1 + pd.Series(acwi_rets)).cumprod().tolist()
            },
            "performance": {
                "equity_engine": get_p_stats(res_rets),
                "acwi": get_p_stats(acwi_rets)
            },
            "weight_history": history
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
