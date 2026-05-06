import math, numpy as np, pandas as pd, yfinance as yf
from flask import Flask, request, jsonify
from flask_cors import CORS
import os, traceback

app = Flask(__name__)
CORS(app)

# --- MASTER ENGINE V4.0 ---

TICKERS_V4 = {
    "equities": "ACWI", "cash": "BIL",
    "sectors": {
        "defense": "ITA", "transport": "IYT", "infra": "IGF", "software": "IGV", 
        "semis": "SMH", "cyber": "HACK", "media": "XLC", "biotech": "XBI", 
        "pharma": "XLV", "medtech": "IHI", "banks": "KBE", "brokers": "IAI", 
        "retail": "XRT", "staples": "XLP", "metals": "XME", "energy": "XLE", 
        "uranium": "URA", "chem": "XLB", "utilities": "XLU", "real_estate": "XLRE"
    }
}

def get_score(prices_slice, cfg, i):
    res = {}
    for col in prices_slice.columns:
        p = prices_slice[col]
        m1 = (p.iloc[i] / p.iloc[i-1] - 1) if i >= 1 else 0
        m3 = (p.iloc[i] / p.iloc[i-3] - 1) if i >= 3 else 0
        m6 = (p.iloc[i] / p.iloc[i-6] - 1) if i >= 6 else 0
        m12 = (p.iloc[i] / p.iloc[i-12] - 1) if i >= 12 else 0
        score = (cfg.get('w1', 0) * m1 + cfg.get('w3', 0) * m3 + 
                 cfg.get('w6', 0) * m6 + cfg.get('w12', 0) * m12)
        res[col] = score
    return res

def simulate_rule(prices_full, rets_full, cfg, start_idx, end_idx, hurdle_y_pa):
    sim_rets = []
    sector_cols = list(TICKERS_V4["sectors"].keys())
    y_monthly = hurdle_y_pa / 12
    for t in range(start_idx, end_idx):
        scores = get_score(prices_full, cfg, t)
        eq_s, cash_s = scores["equities"], scores["cash"]
        qual = {s: sc for s in sector_cols if (sc := scores[s]) > (eq_s + y_monthly) and sc > cash_s}
        if qual:
            top = sorted(qual, key=qual.get, reverse=True)[:5]
            ts = sum(qual[k] for k in top)
            m_ret = sum((qual[k]/ts) * rets_full[k].iloc[t] for k in top)
        elif eq_s > cash_s:
            m_ret = rets_full["equities"].iloc[t]
        else:
            m_ret = rets_full["cash"].iloc[t]
        sim_rets.append(m_ret)
    return (1 + pd.Series(sim_rets)).prod() - 1

@app.route("/api/equity-engine_v4")
def api_v4():
    try:
        start_date = pd.to_datetime(request.args.get("start_date", "2015-01-01"))
        x_win = int(request.args.get("opt_window_x", 36))
        y_pa = float(request.args.get("hurdle_y_pa", 0.02))
        z_lock = int(request.args.get("lock_z", 6))
        
        mapping = TICKERS_V4["sectors"]
        all_t = [TICKERS_V4["equities"], TICKERS_V4["cash"]] + list(mapping.values())
        raw = yf.download(all_t + ["SHV", "XLK"], start="2000-01-01", auto_adjust=True, progress=False)["Close"]
        df = raw.resample("ME").last().ffill()
        
        if "BIL" in df and "SHV" in df: 
            df["BIL"] = df["BIL"].fillna(df["SHV"] * (df["BIL"].dropna().iloc[0]/df["SHV"].loc[df["BIL"].dropna().index[0]]))
        if "XLC" in df and "XLK" in df: 
            df["XLC"] = df["XLC"].fillna(df["XLK"] * (df["XLC"].dropna().iloc[0]/df["XLK"].loc[df["XLC"].dropna().index[0]]))
        
        inv_map = {v: k for k, v in mapping.items()}
        inv_map[TICKERS_V4["equities"]] = "equities"; inv_map[TICKERS_V4["cash"]] = "cash"
        prices = df.rename(columns=inv_map)[[c for c in inv_map.values() if c in df.rename(columns=inv_map).columns]]
        rets_df = prices.pct_change().shift(-1).fillna(0)
        
        configs = [{'w1': 0.1, 'w3': 0.2, 'w6': 0.3, 'w12': 0.4}, {'w6': 0.5, 'w12': 0.5}, {'w3': 0.4, 'w6': 0.6}, {'w12': 1.0}]

        start_i = np.where(prices.index >= start_date)[0][0]
        if start_i < x_win: start_i = x_win
        
        res_rets, acwi_rets, dates, history = [], [], [], []
        curr_cfg, mode, months_active = configs[0], "BETA", 999 

        for i in range(start_i, len(prices)-1):
            if months_active >= z_lock:
                best_cfg, max_alpha = None, -999
                acwi_perf = (1 + prices["equities"].pct_change().iloc[i-x_win:i]).prod() - 1
                for cfg in configs:
                    rule_perf = simulate_rule(prices, rets_df, cfg, i-x_win, i, y_pa)
                    if rule_perf > (acwi_perf + (y_pa * x_win/12)) and rule_perf > max_alpha:
                        max_alpha, best_cfg = rule_perf, cfg
                if best_cfg:
                    curr_cfg, mode, months_active = best_cfg, "ALPHA", 0
                else:
                    mode, months_active = "BETA", 0

            scores = get_score(prices, curr_cfg, i)
            eq_s, cash_s = scores["equities"], scores["cash"]
            qual = {s: sc for s in mapping.keys() if (sc := scores[s]) > (eq_s + y_pa/12) and sc > cash_s}
            
            weights = {}
            if qual and mode == "ALPHA":
                top = sorted(qual, key=qual.get, reverse=True)[:5]
                ts = sum(qual[k] for k in top)
                weights = {k: round(qual[k]/ts, 4) for k in top}
                c_mode = "ALPHA"
            elif eq_s > cash_s:
                weights["equities"] = 1.0; c_mode = "BETA"
            else:
                weights["cash"] = 1.0; c_mode = "CASH"

            res_rets.append(sum(w * rets_df[a].iloc[i] for a, w in weights.items()))
            acwi_rets.append(rets_df["equities"].iloc[i])
            dates.append(prices.index[i+1])
            history.append({"date": prices.index[i].strftime("%Y-%m-%d"), "weights": weights, "mode": c_mode})
            months_active += 1

        def get_p_stats(r_list):
            r_ser = pd.Series(r_list)
            cum = (1 + r_ser).cumprod()
            y = len(r_ser) / 12
            cagr = float(cum.iloc[-1]**(1/y) - 1) if y > 0 else 0
            dd = float((cum / cum.cummax() - 1).min())
            vola = float(r_ser.std() * math.sqrt(12))
            return {"cagr": round(cagr, 4), "max_drawdown": round(dd, 4), "sharpe": round(cagr/vola, 4) if vola > 0 else 0}

        return jsonify({
            "series": {
                "dates": [d.strftime("%Y-%m-%d") for d in dates], 
                "equity_engine": (1 + pd.Series(res_rets)).cumprod().tolist(), 
                "acwi": (1 + pd.Series(acwi_rets)).cumprod().tolist()
            },
            "performance": {"equity_engine": get_p_stats(res_rets), "acwi": get_p_stats(acwi_rets)},
            "weight_history": history
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
