import math, numpy as np, pandas as pd, yfinance as yf
from flask import Flask, request, jsonify
from flask_cors import CORS
import os, traceback

app = Flask(__name__)
CORS(app)

# --- MASTER ENGINE V3.6.0 (Walk-forward rule selection) ---

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

RULE_PERIODS = (3, 6, 9, 12, 18)


def score_asset(prices, cfg, idx, col):
    p_vec = prices[col]
    score = 0.0
    for period in RULE_PERIODS:
        weight = cfg.get(f"w{period}", 0)
        if weight:
            score += weight * (p_vec.iloc[idx] / p_vec.iloc[idx - period] - 1)
    return score


def select_weights_for_rule(prices, cfg, idx, sector_cols, hurdle_y_pm):
    scores = {s: score_asset(prices, cfg, idx, s) for s in sector_cols}
    eq_s = score_asset(prices, cfg, idx, "equities")
    cash_s = score_asset(prices, cfg, idx, "cash")
    qual = {s: sc for s, sc in scores.items() if sc > (eq_s + hurdle_y_pm) and sc > cash_s}

    if qual:
        top_keys = sorted(qual, key=qual.get, reverse=True)[:5]
        sum_scores = sum(qual[k] for k in top_keys)
        weights = {k: round(qual[k] / sum_scores, 4) for k in top_keys}
        return weights, "ALPHA"
    if eq_s > cash_s:
        return {"equities": 1.0}, "BETA"
    return {"cash": 1.0}, "CASH"


def cumulative_return(rets):
    return float((1 + pd.Series(rets)).prod() - 1)


def simulate_rule_window(prices, rets_df, cfg, start_idx, end_idx, sector_cols, hurdle_y_pm):
    rule_rets = []
    for idx in range(start_idx, end_idx):
        weights, _mode = select_weights_for_rule(prices, cfg, idx, sector_cols, hurdle_y_pm)
        rule_rets.append(sum(w * rets_df[a].iloc[idx + 1] for a, w in weights.items()))
    return cumulative_return(rule_rets)


@app.route("/api/equity-engine-v3")
def api_v3():
    try:
        # 1. Parameter & Daten-Setup
        start_date = pd.to_datetime(request.args.get("start_date", "2015-01-01"))
        x_win = int(request.args.get("opt_window_x", 36))
        y_pa = float(request.args.get("hurdle_y_pa", 0.02)) / 12
        z_lock = int(request.args.get("lock_z", 6))
        cost_rate = float(request.args.get("cost_rate", 0.001))
        
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
        
        # Regeln: unterschiedliche Gewichtung der 3/6/9/12/18-Monatsperformance
        configs = [
            {'name': 'Medium (6/9)', 'w6': 0.50, 'w9': 0.50},
            {'name': 'Long (9/12)', 'w9': 0.20, 'w12': 0.80},
            {'name': 'Extra Long (12/18)', 'w12': 0.50, 'w18': 0.50}
        ]

        # 3. Backtest-Loop
        start_i = np.where(prices.index >= start_date)[0][0]
        min_i = max(x_win, max(RULE_PERIODS))
        if start_i < min_i: start_i = min_i
        
        res_rets, acwi_rets, dates, history = [], [], [], []
        curr_p, alpha_lock_remaining = None, 0
        last_weights = {} 
        sector_cols = list(mapping.keys())

        for i in range(start_i, len(prices)-1):
            window_start = i - x_win
            rule_test = None

            if alpha_lock_remaining <= 0:
                rule_results = [
                    {
                        "cfg": cfg,
                        "return": simulate_rule_window(prices, rets_df, cfg, window_start, i, sector_cols, y_pa)
                    }
                    for cfg in configs
                ]
                best_rule = max(rule_results, key=lambda r: r["return"])
                acwi_window_return = cumulative_return(rets_df["equities"].iloc[window_start + 1:i + 1])
                cash_window_return = cumulative_return(rets_df["cash"].iloc[window_start + 1:i + 1])
                rule_test = {
                    "best_rule": best_rule["cfg"]["name"],
                    "best_rule_return": round(best_rule["return"], 4),
                    "acwi_return": round(acwi_window_return, 4),
                    "cash_return": round(cash_window_return, 4)
                }

                if best_rule["return"] > acwi_window_return and best_rule["return"] > cash_window_return:
                    curr_p = best_rule["cfg"]
                    alpha_lock_remaining = z_lock
                else:
                    curr_p = None

            if curr_p is not None:
                weights, mode = select_weights_for_rule(prices, curr_p, i, sector_cols, y_pa)
                rule_name = curr_p["name"]
                alpha_lock_remaining -= 1
            elif rule_test and rule_test["acwi_return"] > rule_test["cash_return"]:
                weights, mode, rule_name = {"equities": 1.0}, "BETA", "N/A"
            else:
                weights, mode, rule_name = {"cash": 1.0}, "CASH", "N/A"

            # Turnover & Kosten
            turnover = sum(abs(weights.get(a, 0) - last_weights.get(a, 0)) for a in set(list(weights.keys()) + list(last_weights.keys())))
            m_cost = turnover * cost_rate
            m_ret = sum(w * rets_df[a].iloc[i+1] for a, w in weights.items()) - m_cost
            
            res_rets.append(m_ret)
            acwi_rets.append(rets_df["equities"].iloc[i+1])
            dates.append(prices.index[i+1])
            
            history.append({
                "date": prices.index[i+1].strftime("%Y-%m-%d"), 
                "weights": weights, 
                "mode": mode,
                "rule": rule_name,
                "rule_test": rule_test
            })
            
            last_weights = weights.copy()

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
            "performance": {"equity_engine": get_p_stats(res_rets), "acwi": get_p_stats(acwi_rets)},
            "weight_history": history
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
