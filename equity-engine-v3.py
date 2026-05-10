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


def load_prices():
    mapping = TICKERS_V3["sectors"]
    all_t = [TICKERS_V3["equities"], TICKERS_V3["cash"]] + list(mapping.values())

    raw = yf.download(all_t + ["SHV", "XLK", "XLI"], start="2000-01-01", auto_adjust=True, progress=False)["Close"]
    df = raw.resample("ME").last().ffill()

    if "BIL" in df and "SHV" in df:
        df["BIL"] = df["BIL"].fillna(df["SHV"] * (df["BIL"].dropna().iloc[0] / df["SHV"].loc[df["BIL"].dropna().index[0]]))
    if "XLC" in df and "XLK" in df:
        df["XLC"] = df["XLC"].fillna(df["XLK"] * (df["XLC"].dropna().iloc[0] / df["XLK"].loc[df["XLK"].dropna().index[0]]))

    inv_map = {v: k for k, v in mapping.items()}
    inv_map[TICKERS_V3["equities"]] = "equities"
    inv_map[TICKERS_V3["cash"]] = "cash"
    prices = df.rename(columns=inv_map)[[c for c in inv_map.values() if c in df.rename(columns=inv_map).columns]]
    return prices, prices.pct_change().fillna(0), list(mapping.keys())


def rule_configs():
    return [
        {'name': 'Medium (6/9)', 'w6': 0.50, 'w9': 0.50},
        {'name': 'Long (9/12)', 'w9': 0.20, 'w12': 0.80},
        {'name': 'Extra Long (12/18)', 'w12': 0.50, 'w18': 0.50}
    ]


def get_p_stats(r_list):
    r_ser = pd.Series(r_list)
    if r_ser.empty:
        return {"cagr": 0, "max_drawdown": 0, "sharpe": 0}
    cum = (1 + r_ser).cumprod()
    y = len(r_ser) / 12
    cagr = float(cum.iloc[-1]**(1/y) - 1) if y > 0 else 0
    dd = float((cum / cum.cummax() - 1).min())
    vola = float(r_ser.std() * math.sqrt(12))
    return {"cagr": round(cagr, 4), "max_drawdown": round(dd, 4), "sharpe": round(cagr / vola, 4) if vola > 0 else 0}


def run_backtest(start_date, x_win, y_pa, z_lock, cost_rate):
    prices, rets_df, sector_cols = load_prices()
    configs = rule_configs()
    always_long_cfg = {'name': 'Always Long (9/12)', 'w9': 0.20, 'w12': 0.80}

    start_i = np.where(prices.index >= start_date)[0][0]
    min_i = max(x_win, max(RULE_PERIODS))
    if start_i < min_i:
        start_i = min_i

    res_rets, long_rets, acwi_rets, dates, history = [], [], [], [], []
    curr_p, alpha_lock_remaining = None, 0
    last_weights = {}
    last_long_weights = {}

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

        turnover = sum(abs(weights.get(a, 0) - last_weights.get(a, 0)) for a in set(list(weights.keys()) + list(last_weights.keys())))
        m_ret = sum(w * rets_df[a].iloc[i+1] for a, w in weights.items()) - (turnover * cost_rate)

        long_weights, _long_mode = select_weights_for_rule(prices, always_long_cfg, i, sector_cols, y_pa)
        long_turnover = sum(abs(long_weights.get(a, 0) - last_long_weights.get(a, 0)) for a in set(list(long_weights.keys()) + list(last_long_weights.keys())))
        long_m_ret = sum(w * rets_df[a].iloc[i+1] for a, w in long_weights.items()) - (long_turnover * cost_rate)

        res_rets.append(m_ret)
        long_rets.append(long_m_ret)
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
        last_long_weights = long_weights.copy()

    return {
        "dates": dates,
        "returns": {
            "equity_engine": res_rets,
            "always_long": long_rets,
            "acwi": acwi_rets
        },
        "weight_history": history
    }


def backtest_response(result):
    return {
        "series": {
            "dates": [d.strftime("%Y-%m-%d") for d in result["dates"]],
            "equity_engine": (1 + pd.Series(result["returns"]["equity_engine"])).cumprod().tolist(),
            "always_long": (1 + pd.Series(result["returns"]["always_long"])).cumprod().tolist(),
            "acwi": (1 + pd.Series(result["returns"]["acwi"])).cumprod().tolist()
        },
        "performance": {
            "equity_engine": get_p_stats(result["returns"]["equity_engine"]),
            "always_long": get_p_stats(result["returns"]["always_long"]),
            "acwi": get_p_stats(result["returns"]["acwi"])
        },
        "weight_history": result["weight_history"]
    }


def annualized_return(rets):
    if not rets:
        return 0
    return float((1 + pd.Series(rets)).prod() ** (12 / len(rets)) - 1)


def build_research(result, horizons):
    strategies = ["equity_engine", "always_long", "acwi"]
    windows = []
    summary = {}

    for horizon in horizons:
        rows = []
        for start in range(0, len(result["dates"]) - horizon + 1):
            end = start + horizon
            metrics = {}
            for strategy in strategies:
                window_rets = result["returns"][strategy][start:end]
                metrics[strategy] = {
                    "cagr": round(annualized_return(window_rets), 4),
                    "max_drawdown": get_p_stats(window_rets)["max_drawdown"],
                    "sharpe": get_p_stats(window_rets)["sharpe"]
                }
            winner = max(strategies, key=lambda s: metrics[s]["cagr"])
            rows.append({
                "start": result["dates"][start].strftime("%Y-%m-%d"),
                "end": result["dates"][end - 1].strftime("%Y-%m-%d"),
                "winner": winner,
                "metrics": metrics
            })

        windows.append({"horizon_months": horizon, "windows": rows})
        summary[str(horizon)] = {}
        for strategy in strategies:
            if strategy == "acwi":
                continue
            alphas = [r["metrics"][strategy]["cagr"] - r["metrics"]["acwi"]["cagr"] for r in rows]
            summary[str(horizon)][strategy] = {
                "windows": len(rows),
                "win_rate_vs_acwi": round(sum(1 for a in alphas if a > 0) / len(alphas), 4) if alphas else 0,
                "avg_alpha_pa": round(float(np.mean(alphas)), 4) if alphas else 0,
                "median_alpha_pa": round(float(np.median(alphas)), 4) if alphas else 0,
                "worst_alpha_pa": round(float(np.min(alphas)), 4) if alphas else 0,
                "best_alpha_pa": round(float(np.max(alphas)), 4) if alphas else 0
            }

    return {"summary": summary, "rolling_windows": windows}


def simulate_fixed_rule(prices, rets_df, cfg, start_date, x_win, cost_rate, hurdle_y_pm, sector_cols):
    start_i = np.where(prices.index >= start_date)[0][0]
    min_i = max(x_win, max(RULE_PERIODS))
    if start_i < min_i:
        start_i = min_i

    strategy_rets, acwi_rets, dates = [], [], []
    last_weights = {}
    for i in range(start_i, len(prices)-1):
        weights, _mode = select_weights_for_rule(prices, cfg, i, sector_cols, hurdle_y_pm)
        turnover = sum(abs(weights.get(a, 0) - last_weights.get(a, 0)) for a in set(list(weights.keys()) + list(last_weights.keys())))
        strategy_rets.append(sum(w * rets_df[a].iloc[i+1] for a, w in weights.items()) - (turnover * cost_rate))
        acwi_rets.append(rets_df["equities"].iloc[i+1])
        dates.append(prices.index[i+1])
        last_weights = weights.copy()

    return {"dates": dates, "returns": {"strategy": strategy_rets, "acwi": acwi_rets}}


def rolling_alpha_summary(dates, strategy_rets, acwi_rets, horizons):
    summary = {}
    for horizon in horizons:
        alphas = []
        for start in range(0, len(dates) - horizon + 1):
            end = start + horizon
            strategy_cagr = annualized_return(strategy_rets[start:end])
            acwi_cagr = annualized_return(acwi_rets[start:end])
            alphas.append(strategy_cagr - acwi_cagr)
        summary[str(horizon)] = {
            "windows": len(alphas),
            "win_rate_vs_acwi": round(sum(1 for a in alphas if a > 0) / len(alphas), 4) if alphas else 0,
            "avg_alpha_pa": round(float(np.mean(alphas)), 4) if alphas else 0,
            "median_alpha_pa": round(float(np.median(alphas)), 4) if alphas else 0,
            "worst_alpha_pa": round(float(np.min(alphas)), 4) if alphas else 0,
            "best_alpha_pa": round(float(np.max(alphas)), 4) if alphas else 0
        }
    return summary


@app.route("/api/equity-engine-v3")
def api_v3():
    try:
        start_date = pd.to_datetime(request.args.get("start_date", "2015-01-01"))
        x_win = int(request.args.get("opt_window_x", 36))
        y_pa = float(request.args.get("hurdle_y_pa", 0.02)) / 12
        z_lock = int(request.args.get("lock_z", 6))
        cost_rate = float(request.args.get("cost_rate", 0.001))
        return jsonify(backtest_response(run_backtest(start_date, x_win, y_pa, z_lock, cost_rate)))
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/equity-engine-v3/rule-grid")
def api_v3_rule_grid():
    try:
        start_date = pd.to_datetime(request.args.get("start_date", "2011-01-01"))
        x_win = int(request.args.get("opt_window_x", 36))
        y_pa = float(request.args.get("hurdle_y_pa", 0.02)) / 12
        cost_rate = float(request.args.get("cost_rate", 0.001))
        horizons = [int(h) for h in request.args.get("horizons", "12,36,60,120").split(",") if h.strip()]
        step = int(request.args.get("step", 10))

        prices, rets_df, sector_cols = load_prices()
        results = []
        for w9_pct in range(0, 101, step):
            w12_pct = 100 - w9_pct
            cfg = {
                "name": f"Fixed 9/12 ({w9_pct}/{w12_pct})",
                "w9": w9_pct / 100,
                "w12": w12_pct / 100
            }
            sim = simulate_fixed_rule(prices, rets_df, cfg, start_date, x_win, cost_rate, y_pa, sector_cols)
            strategy_rets = sim["returns"]["strategy"]
            acwi_rets = sim["returns"]["acwi"]
            perf = get_p_stats(strategy_rets)
            acwi_perf = get_p_stats(acwi_rets)
            rolling = rolling_alpha_summary(sim["dates"], strategy_rets, acwi_rets, horizons)
            results.append({
                "rule": cfg["name"],
                "weights": {"9m": cfg["w9"], "12m": cfg["w12"]},
                "performance": {
                    "strategy": perf,
                    "acwi": acwi_perf,
                    "alpha_cagr": round(perf["cagr"] - acwi_perf["cagr"], 4)
                },
                "rolling": rolling
            })

        results.sort(key=lambda r: (r["rolling"].get("60", {}).get("win_rate_vs_acwi", 0), r["performance"]["strategy"]["sharpe"], r["performance"]["strategy"]["cagr"]), reverse=True)
        return jsonify({
            "params": {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "opt_window_x": x_win,
                "hurdle_y_pa": round(y_pa * 12, 4),
                "cost_rate": cost_rate,
                "horizons": horizons,
                "step": step,
                "grid": "9m/12m"
            },
            "results": results
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/equity-engine-v3/research")
def api_v3_research():
    try:
        start_date = pd.to_datetime(request.args.get("start_date", "2011-01-01"))
        x_win = int(request.args.get("opt_window_x", 36))
        y_pa = float(request.args.get("hurdle_y_pa", 0.02)) / 12
        z_lock = int(request.args.get("lock_z", 6))
        cost_rate = float(request.args.get("cost_rate", 0.001))
        horizons = [int(h) for h in request.args.get("horizons", "12,36,60,120").split(",") if h.strip()]
        result = run_backtest(start_date, x_win, y_pa, z_lock, cost_rate)
        return jsonify({
            "params": {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "opt_window_x": x_win,
                "hurdle_y_pa": round(y_pa * 12, 4),
                "lock_z": z_lock,
                "cost_rate": cost_rate,
                "horizons": horizons
            },
            **build_research(result, horizons)
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
