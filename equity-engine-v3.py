import json, math, time, numpy as np, pandas as pd, yfinance as yf
from flask import Flask, request, jsonify
from flask_cors import CORS
from itertools import combinations
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
        "uranium": "URA", "chem": "XLB", "utilities": "XLU", "real_estate": "XLRE",
        "copper_miners": "COPX", "gold_miners": "GDX", "us_infra": "PAVE",
        "consumer_discretionary": "XLY", "emerging_markets": "EEM",
        "europe": "VGK", "rare_earths": "REMX", "lithium_battery": "LIT",
        "robotics_ai": "BOTZ", "solar": "TAN", "natural_resources": "GUNR",
        "semiconductors": "SOXX", "gold": "GLD"
    }
}

RULE_PERIODS = (3, 6, 9, 12, 18)
PRICE_START_DATE = "2000-01-01"
PRICE_CACHE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60
PRICE_CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")
PRICE_CACHE_FILE = os.path.join(PRICE_CACHE_DIR, "equity_engine_v3_prices.pkl")
PRICE_CACHE_META_FILE = os.path.join(PRICE_CACHE_DIR, "equity_engine_v3_prices.json")
PRICE_FALLBACK_TICKERS = ("SHV", "XLK", "XLI")


def score_asset(prices, cfg, idx, col):
    p_vec = prices[col]
    score = 0.0
    for period in RULE_PERIODS:
        weight = cfg.get(f"w{period}", 0)
        if weight:
            current = p_vec.iloc[idx]
            previous = p_vec.iloc[idx - period]
            if not np.isfinite(current) or not np.isfinite(previous) or previous <= 0:
                return -np.inf
            score += weight * (current / previous - 1)
    return score


def select_weights_for_rule(prices, cfg, idx, sector_cols, hurdle_y_pm):
    scores = {s: score_asset(prices, cfg, idx, s) for s in sector_cols}
    eq_s = score_asset(prices, cfg, idx, "equities")
    cash_s = score_asset(prices, cfg, idx, "cash")
    qual = {s: sc for s, sc in scores.items() if sc > (eq_s + hurdle_y_pm) and sc > cash_s}

    if qual:
        top_keys = sorted(qual, key=qual.get, reverse=True)[:5]
        sum_scores = sum(qual[k] for k in top_keys)
        weights = {k: float(round(qual[k] / sum_scores, 4)) for k in top_keys}
        return weights, "ALPHA"
    if eq_s > cash_s:
        return {"equities": 1.0}, "BETA"
    return {"cash": 1.0}, "CASH"


def frog_score_asset(prices, rets_df, cfg, idx, col, months_back=18):
    hits, total = 0, 0
    start = max(1, idx - months_back + 1)
    for ret_idx in range(start, idx + 1):
        signal_idx = ret_idx - 1
        if signal_idx < max(RULE_PERIODS):
            continue
        signal_score = score_asset(prices, cfg, signal_idx, col)
        ret = rets_df[col].iloc[ret_idx]
        if not np.isfinite(signal_score) or not np.isfinite(ret):
            continue
        if signal_score > 0:
            total += 1
            if ret > 0:
                hits += 1
    return float(hits) / float(total) if total else 0.0


def select_weights_for_rule_with_frog(prices, rets_df, cfg, idx, sector_cols, hurdle_y_pm, frog_min_score):
    scores = {s: score_asset(prices, cfg, idx, s) for s in sector_cols}
    frog_scores = {s: frog_score_asset(prices, rets_df, cfg, idx, s) for s in sector_cols}
    eq_s = score_asset(prices, cfg, idx, "equities")
    cash_s = score_asset(prices, cfg, idx, "cash")
    qual = {
        s: sc for s, sc in scores.items()
        if sc > (eq_s + hurdle_y_pm) and sc > cash_s and frog_scores.get(s, 0) >= frog_min_score
    }

    if qual:
        top_keys = sorted(qual, key=qual.get, reverse=True)[:5]
        sum_scores = sum(qual[k] for k in top_keys)
        if sum_scores <= 0:
            return {"equities": 1.0}, "BETA"
        weights = {k: float(round(qual[k] / sum_scores, 4)) for k in top_keys}
        return weights, "FROG"
    if eq_s > cash_s:
        return {"equities": 1.0}, "BETA"
    return {"cash": 1.0}, "CASH"


def signal_return_ev(prices, rets_df, cfg, idx, col, months_back=36):
    signal_rets = []
    start = max(1, idx - months_back + 1)
    for ret_idx in range(start, idx + 1):
        signal_idx = ret_idx - 1
        if signal_idx < max(RULE_PERIODS):
            continue
        signal_score = score_asset(prices, cfg, signal_idx, col)
        ret = rets_df[col].iloc[ret_idx]
        if np.isfinite(signal_score) and signal_score > 0 and np.isfinite(ret):
            signal_rets.append(float(ret))
    if not signal_rets:
        return -np.inf
    return float(np.mean(signal_rets))


def ev_utility_score(prices, rets_df, cfg, idx, col):
    momentum_score = score_asset(prices, cfg, idx, col)
    signal_ev = signal_return_ev(prices, rets_df, cfg, idx, col)
    if not np.isfinite(momentum_score) or momentum_score <= 0 or not np.isfinite(signal_ev):
        return -np.inf
    return float(momentum_score * signal_ev)


def select_weights_for_rule_with_ev(prices, rets_df, cfg, idx, sector_cols, hurdle_y_pm, last_weights, switch_threshold):
    momentum_scores = {s: score_asset(prices, cfg, idx, s) for s in sector_cols}
    utility_scores = {s: ev_utility_score(prices, rets_df, cfg, idx, s) for s in sector_cols}
    eq_s = score_asset(prices, cfg, idx, "equities")
    cash_s = score_asset(prices, cfg, idx, "cash")
    qual = {
        s: utility_scores[s] for s in sector_cols
        if momentum_scores[s] > (eq_s + hurdle_y_pm)
        and momentum_scores[s] > cash_s
        and utility_scores[s] > 0
    }

    if qual:
        top_keys = sorted(qual, key=qual.get, reverse=True)[:5]
        sum_scores = sum(qual[k] for k in top_keys)
        candidate_weights = {k: float(round(qual[k] / sum_scores, 4)) for k in top_keys}
        candidate_mode = "EV"
    elif eq_s > cash_s:
        candidate_weights, candidate_mode = {"equities": 1.0}, "BETA"
    else:
        candidate_weights, candidate_mode = {"cash": 1.0}, "CASH"

    if last_weights:
        candidate_ev = sum(candidate_weights.get(a, 0) * utility_scores.get(a, 0) for a in candidate_weights)
        current_ev = sum(last_weights.get(a, 0) * utility_scores.get(a, 0) for a in last_weights)
        if np.isfinite(candidate_ev) and np.isfinite(current_ev) and (candidate_ev - current_ev) <= switch_threshold:
            return last_weights.copy(), "HOLD"

    return candidate_weights, candidate_mode


def cumulative_return(rets):
    return float((1 + pd.Series(rets)).prod() - 1)


def simulate_rule_window(prices, rets_df, cfg, start_idx, end_idx, sector_cols, hurdle_y_pm):
    rule_rets = []
    for idx in range(start_idx, end_idx):
        weights, _mode = select_weights_for_rule(prices, cfg, idx, sector_cols, hurdle_y_pm)
        rule_rets.append(sum(w * rets_df[a].iloc[idx + 1] for a, w in weights.items()))
    return cumulative_return(rule_rets)


def read_price_cache(tickers, start, max_age_seconds=PRICE_CACHE_MAX_AGE_SECONDS):
    if not os.path.exists(PRICE_CACHE_FILE) or not os.path.exists(PRICE_CACHE_META_FILE):
        return None
    try:
        with open(PRICE_CACHE_META_FILE, "r", encoding="utf-8") as meta_file:
            meta = json.load(meta_file)
        if meta.get("tickers") != sorted(tickers) or meta.get("start") != start:
            return None
        fetched_at = float(meta.get("fetched_at", 0))
        if max_age_seconds is not None and (time.time() - fetched_at) > max_age_seconds:
            return None
        prices = pd.read_pickle(PRICE_CACHE_FILE)
        if isinstance(prices, pd.DataFrame) and not prices.empty:
            return prices
    except Exception as exc:
        print(f"Price cache ignored: {exc}")
    return None


def write_price_cache(prices, tickers, start):
    os.makedirs(PRICE_CACHE_DIR, exist_ok=True)
    prices.to_pickle(PRICE_CACHE_FILE)
    meta = {
        "tickers": sorted(tickers),
        "start": start,
        "fetched_at": time.time(),
        "max_age_days": PRICE_CACHE_MAX_AGE_SECONDS // (24 * 60 * 60),
    }
    with open(PRICE_CACHE_META_FILE, "w", encoding="utf-8") as meta_file:
        json.dump(meta, meta_file, indent=2, sort_keys=True)


def download_price_data(tickers, start):
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame()
    if raw.empty:
        raise ValueError("No price data could be loaded from yfinance.")
    return raw


def load_cached_price_data(tickers, start):
    cached = read_price_cache(tickers, start)
    if cached is not None:
        print("Using cached price data.")
        return cached

    try:
        fresh = download_price_data(tickers, start)
        write_price_cache(fresh, tickers, start)
        print("Downloaded fresh price data and updated cache.")
        return fresh
    except Exception:
        stale = read_price_cache(tickers, start, max_age_seconds=None)
        if stale is not None:
            print("Using stale cached price data because fresh download failed.")
            return stale
        raise


def load_prices():
    mapping = TICKERS_V3["sectors"]
    all_t = [TICKERS_V3["equities"], TICKERS_V3["cash"]] + list(mapping.values())
    price_tickers = all_t + list(PRICE_FALLBACK_TICKERS)

    raw = load_cached_price_data(price_tickers, PRICE_START_DATE)
    df = raw.resample("ME").last().ffill()

    if "BIL" in df and "SHV" in df and not df["BIL"].dropna().empty and not df["SHV"].dropna().empty:
        first_bil_idx = df["BIL"].dropna().index[0]
        if first_bil_idx in df["SHV"].dropna().index:
            df["BIL"] = df["BIL"].fillna(df["SHV"] * (df["BIL"].dropna().iloc[0] / df["SHV"].loc[first_bil_idx]))
    if "XLC" in df and "XLK" in df and not df["XLC"].dropna().empty and not df["XLK"].dropna().empty:
        first_xlc_idx = df["XLC"].dropna().index[0]
        if first_xlc_idx in df["XLK"].dropna().index:
            df["XLC"] = df["XLC"].fillna(df["XLK"] * (df["XLC"].dropna().iloc[0] / df["XLK"].loc[first_xlc_idx]))

    inv_map = {v: k for k, v in mapping.items()}
    inv_map[TICKERS_V3["equities"]] = "equities"
    inv_map[TICKERS_V3["cash"]] = "cash"
    prices = df.rename(columns=inv_map)[[c for c in inv_map.values() if c in df.rename(columns=inv_map).columns]]
    if "equities" not in prices or prices["equities"].dropna().empty:
        raise ValueError("ACWI data could not be loaded.")
    if "cash" not in prices or prices["cash"].dropna().empty:
        raise ValueError("BIL data could not be loaded.")
    sector_cols = [sector for sector in mapping.keys() if sector in prices.columns]
    return prices, prices.pct_change().fillna(0), sector_cols


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


def run_backtest(start_date, x_win, y_pa, z_lock, cost_rate, frog_min_score=0.5, ev_switch_threshold=0.002):
    prices, rets_df, sector_cols = load_prices()
    configs = rule_configs()
    fixed_20_80_cfg = {'name': 'Fixed 20/80 (9/12)', 'w9': 0.20, 'w12': 0.80}
    fixed_30_70_cfg = {'name': 'Fixed 30/70 (9/12)', 'w9': 0.30, 'w12': 0.70}

    start_i = np.where(prices.index >= start_date)[0][0]
    min_i = max(x_win, max(RULE_PERIODS))
    if start_i < min_i:
        start_i = min_i

    res_rets, fixed_20_80_rets, fixed_30_70_rets, fixed_30_70_frog_rets, fixed_30_70_ev_rets, acwi_rets, dates, history = [], [], [], [], [], [], [], []
    fixed_30_70_history = []
    fixed_30_70_frog_history = []
    fixed_30_70_ev_history = []
    curr_p, alpha_lock_remaining = None, 0
    last_weights = {}
    last_fixed_20_80_weights = {}
    last_fixed_30_70_weights = {}
    last_fixed_30_70_frog_weights = {}
    last_fixed_30_70_ev_weights = {}

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

        fixed_20_80_weights, _fixed_20_80_mode = select_weights_for_rule(prices, fixed_20_80_cfg, i, sector_cols, y_pa)
        fixed_20_80_turnover = sum(abs(fixed_20_80_weights.get(a, 0) - last_fixed_20_80_weights.get(a, 0)) for a in set(list(fixed_20_80_weights.keys()) + list(last_fixed_20_80_weights.keys())))
        fixed_20_80_m_ret = sum(w * rets_df[a].iloc[i+1] for a, w in fixed_20_80_weights.items()) - (fixed_20_80_turnover * cost_rate)

        fixed_30_70_weights, _fixed_30_70_mode = select_weights_for_rule(prices, fixed_30_70_cfg, i, sector_cols, y_pa)
        fixed_30_70_turnover = sum(abs(fixed_30_70_weights.get(a, 0) - last_fixed_30_70_weights.get(a, 0)) for a in set(list(fixed_30_70_weights.keys()) + list(last_fixed_30_70_weights.keys())))
        fixed_30_70_m_ret = sum(w * rets_df[a].iloc[i+1] for a, w in fixed_30_70_weights.items()) - (fixed_30_70_turnover * cost_rate)

        fixed_30_70_frog_weights, _fixed_30_70_frog_mode = select_weights_for_rule_with_frog(prices, rets_df, fixed_30_70_cfg, i, sector_cols, y_pa, frog_min_score)
        fixed_30_70_frog_turnover = sum(abs(fixed_30_70_frog_weights.get(a, 0) - last_fixed_30_70_frog_weights.get(a, 0)) for a in set(list(fixed_30_70_frog_weights.keys()) + list(last_fixed_30_70_frog_weights.keys())))
        fixed_30_70_frog_m_ret = sum(w * rets_df[a].iloc[i+1] for a, w in fixed_30_70_frog_weights.items()) - (fixed_30_70_frog_turnover * cost_rate)

        fixed_30_70_ev_weights, _fixed_30_70_ev_mode = select_weights_for_rule_with_ev(prices, rets_df, fixed_30_70_cfg, i, sector_cols, y_pa, last_fixed_30_70_ev_weights, ev_switch_threshold)
        fixed_30_70_ev_turnover = sum(abs(fixed_30_70_ev_weights.get(a, 0) - last_fixed_30_70_ev_weights.get(a, 0)) for a in set(list(fixed_30_70_ev_weights.keys()) + list(last_fixed_30_70_ev_weights.keys())))
        fixed_30_70_ev_m_ret = sum(w * rets_df[a].iloc[i+1] for a, w in fixed_30_70_ev_weights.items()) - (fixed_30_70_ev_turnover * cost_rate)

        res_rets.append(m_ret)
        fixed_20_80_rets.append(fixed_20_80_m_ret)
        fixed_30_70_rets.append(fixed_30_70_m_ret)
        fixed_30_70_frog_rets.append(fixed_30_70_frog_m_ret)
        fixed_30_70_ev_rets.append(fixed_30_70_ev_m_ret)
        acwi_rets.append(rets_df["equities"].iloc[i+1])
        dates.append(prices.index[i+1])

        history.append({
            "date": prices.index[i+1].strftime("%Y-%m-%d"),
            "weights": weights,
            "mode": mode,
            "rule": rule_name,
            "rule_test": rule_test
        })
        fixed_30_70_history.append({
            "date": prices.index[i+1].strftime("%Y-%m-%d"),
            "weights": fixed_30_70_weights,
            "mode": _fixed_30_70_mode,
            "rule": fixed_30_70_cfg["name"]
        })
        fixed_30_70_frog_history.append({
            "date": prices.index[i+1].strftime("%Y-%m-%d"),
            "weights": fixed_30_70_frog_weights,
            "mode": _fixed_30_70_frog_mode,
            "rule": "Fixed 30/70 + Frosch"
        })
        fixed_30_70_ev_history.append({
            "date": prices.index[i+1].strftime("%Y-%m-%d"),
            "weights": fixed_30_70_ev_weights,
            "mode": _fixed_30_70_ev_mode,
            "rule": "Fixed 30/70 EV"
        })

        last_weights = weights.copy()
        last_fixed_20_80_weights = fixed_20_80_weights.copy()
        last_fixed_30_70_weights = fixed_30_70_weights.copy()
        last_fixed_30_70_frog_weights = fixed_30_70_frog_weights.copy()
        last_fixed_30_70_ev_weights = fixed_30_70_ev_weights.copy()

    return {
        "dates": dates,
        "returns": {
            "equity_engine": res_rets,
            "always_long": fixed_20_80_rets,
            "fixed_20_80": fixed_20_80_rets,
            "fixed_30_70": fixed_30_70_rets,
            "fixed_30_70_frog": fixed_30_70_frog_rets,
            "fixed_30_70_ev": fixed_30_70_ev_rets,
            "acwi": acwi_rets
        },
        "weight_history": history,
        "fixed_30_70_history": fixed_30_70_history,
        "fixed_30_70_frog_history": fixed_30_70_frog_history,
        "fixed_30_70_ev_history": fixed_30_70_ev_history
    }


def backtest_response(result):
    asset_tickers = {
        "equities": TICKERS_V3["equities"],
        "cash": TICKERS_V3["cash"],
        **TICKERS_V3["sectors"]
    }
    return {
        "series": {
            "dates": [d.strftime("%Y-%m-%d") for d in result["dates"]],
            "equity_engine": (1 + pd.Series(result["returns"]["equity_engine"])).cumprod().tolist(),
            "always_long": (1 + pd.Series(result["returns"]["always_long"])).cumprod().tolist(),
            "fixed_20_80": (1 + pd.Series(result["returns"]["fixed_20_80"])).cumprod().tolist(),
            "fixed_30_70": (1 + pd.Series(result["returns"]["fixed_30_70"])).cumprod().tolist(),
            "fixed_30_70_frog": (1 + pd.Series(result["returns"]["fixed_30_70_frog"])).cumprod().tolist(),
            "fixed_30_70_ev": (1 + pd.Series(result["returns"]["fixed_30_70_ev"])).cumprod().tolist(),
            "acwi": (1 + pd.Series(result["returns"]["acwi"])).cumprod().tolist()
        },
        "performance": {
            "equity_engine": get_p_stats(result["returns"]["equity_engine"]),
            "always_long": get_p_stats(result["returns"]["always_long"]),
            "fixed_20_80": get_p_stats(result["returns"]["fixed_20_80"]),
            "fixed_30_70": get_p_stats(result["returns"]["fixed_30_70"]),
            "fixed_30_70_frog": get_p_stats(result["returns"]["fixed_30_70_frog"]),
            "fixed_30_70_ev": get_p_stats(result["returns"]["fixed_30_70_ev"]),
            "acwi": get_p_stats(result["returns"]["acwi"])
        },
        "weight_history": result["weight_history"],
        "fixed_30_70_history": result["fixed_30_70_history"],
        "fixed_30_70_frog_history": result["fixed_30_70_frog_history"],
        "fixed_30_70_ev_history": result["fixed_30_70_ev_history"],
        "asset_tickers": asset_tickers
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


def build_rule_grid_configs(step):
    configs = []
    for w9_pct in range(0, 101, step):
        w12_pct = 100 - w9_pct
        configs.append({
            "name": f"Fixed 9/12 ({w9_pct}/{w12_pct})",
            "w9": w9_pct / 100,
            "w12": w12_pct / 100
        })
    return configs


def rule_name_from_ratio(ratio):
    left, right = ratio.split("/")
    return f"Fixed 9/12 ({int(left)}/{int(right)})"


@app.route("/api/equity-engine-v3")
def api_v3():
    try:
        start_date = pd.to_datetime(request.args.get("start_date", "2015-01-01"))
        x_win = int(request.args.get("opt_window_x", 36))
        y_pa = float(request.args.get("hurdle_y_pa", 0.02)) / 12
        z_lock = int(request.args.get("lock_z", 6))
        cost_rate = float(request.args.get("cost_rate", 0.001))
        frog_min_score = float(request.args.get("frog_min_score", 0.5))
        ev_switch_threshold = float(request.args.get("ev_switch_threshold", 0.002))
        return jsonify(backtest_response(run_backtest(start_date, x_win, y_pa, z_lock, cost_rate, frog_min_score, ev_switch_threshold)))
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
        for cfg in build_rule_grid_configs(step):
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


@app.route("/api/equity-engine-v3/rule-coverage")
def api_v3_rule_coverage():
    try:
        start_date = pd.to_datetime(request.args.get("start_date", "2011-01-01"))
        x_win = int(request.args.get("opt_window_x", 36))
        y_pa = float(request.args.get("hurdle_y_pa", 0.02)) / 12
        cost_rate = float(request.args.get("cost_rate", 0.001))
        step = int(request.args.get("step", 10))
        max_rules = int(request.args.get("max_rules", 3))
        target_rules = [rule_name_from_ratio(r.strip()) for r in request.args.get("target_rules", "20/80,30/70").split(",") if r.strip()]

        prices, rets_df, sector_cols = load_prices()
        rule_sims = {}
        dates, acwi_rets = None, None
        for cfg in build_rule_grid_configs(step):
            sim = simulate_fixed_rule(prices, rets_df, cfg, start_date, x_win, cost_rate, y_pa, sector_cols)
            dates = sim["dates"]
            acwi_rets = sim["returns"]["acwi"]
            rule_sims[cfg["name"]] = {
                "weights": {"9m": cfg["w9"], "12m": cfg["w12"]},
                "returns": sim["returns"]["strategy"]
            }

        rules = list(rule_sims.keys())
        month_rows = []
        for idx, date in enumerate(dates):
            alphas = {rule: rule_sims[rule]["returns"][idx] - acwi_rets[idx] for rule in rules}
            winners = [rule for rule, alpha in alphas.items() if alpha > 0]
            best_rule = max(rules, key=lambda rule: alphas[rule])
            month_rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "acwi_return": round(acwi_rets[idx], 4),
                "covered_by_any_rule": bool(winners),
                "winning_rules": winners,
                "best_rule": best_rule,
                "best_alpha": round(alphas[best_rule], 4)
            })

        uncovered = [row for row in month_rows if not row["covered_by_any_rule"]]
        single_rule_stats = []
        for rule in rules:
            alphas = [rule_sims[rule]["returns"][i] - acwi_rets[i] for i in range(len(dates))]
            single_rule_stats.append({
                "rule": rule,
                "weights": rule_sims[rule]["weights"],
                "months_beating_acwi": sum(1 for a in alphas if a > 0),
                "coverage_rate": round(sum(1 for a in alphas if a > 0) / len(alphas), 4) if alphas else 0,
                "avg_monthly_alpha": round(float(np.mean(alphas)), 4) if alphas else 0,
                "worst_monthly_alpha": round(float(np.min(alphas)), 4) if alphas else 0
            })
        single_rule_stats.sort(key=lambda r: (r["coverage_rate"], r["avg_monthly_alpha"]), reverse=True)

        target_rule_details = {}
        for rule in target_rules:
            if rule not in rule_sims:
                continue
            alphas = [rule_sims[rule]["returns"][i] - acwi_rets[i] for i in range(len(dates))]
            misses = []
            for idx, alpha in enumerate(alphas):
                if alpha < 0:
                    best_rule = max(rules, key=lambda candidate: rule_sims[candidate]["returns"][idx] - acwi_rets[idx])
                    best_alpha = rule_sims[best_rule]["returns"][idx] - acwi_rets[idx]
                    misses.append({
                        "date": dates[idx].strftime("%Y-%m-%d"),
                        "year": int(dates[idx].year),
                        "month": int(dates[idx].month),
                        "strategy_return": round(rule_sims[rule]["returns"][idx], 4),
                        "acwi_return": round(acwi_rets[idx], 4),
                        "alpha": round(alpha, 4),
                        "best_alternative_rule": best_rule,
                        "best_alternative_alpha": round(best_alpha, 4)
                    })

            by_year, by_month = {}, {}
            for miss in misses:
                by_year[miss["year"]] = by_year.get(miss["year"], 0) + 1
                by_month[miss["month"]] = by_month.get(miss["month"], 0) + 1

            target_rule_details[rule] = {
                "weights": rule_sims[rule]["weights"],
                "missed_months": len(misses),
                "hit_months": len(dates) - len(misses),
                "hit_rate": round((len(dates) - len(misses)) / len(dates), 4) if dates else 0,
                "avg_miss_alpha": round(float(np.mean([m["alpha"] for m in misses])), 4) if misses else 0,
                "median_miss_alpha": round(float(np.median([m["alpha"] for m in misses])), 4) if misses else 0,
                "worst_miss_alpha": round(float(np.min([m["alpha"] for m in misses])), 4) if misses else 0,
                "best_miss_alpha": round(float(np.max([m["alpha"] for m in misses])), 4) if misses else 0,
                "misses_by_year": dict(sorted(by_year.items())),
                "misses_by_calendar_month": dict(sorted(by_month.items())),
                "misses": misses
            }

        combo_stats = []
        for combo_size in range(1, min(max_rules, len(rules)) + 1):
            for combo in combinations(rules, combo_size):
                best_combo_alphas = [
                    max(rule_sims[rule]["returns"][i] - acwi_rets[i] for rule in combo)
                    for i in range(len(dates))
                ]
                combo_stats.append({
                    "rules": list(combo),
                    "months_beating_acwi": sum(1 for a in best_combo_alphas if a > 0),
                    "coverage_rate": round(sum(1 for a in best_combo_alphas if a > 0) / len(best_combo_alphas), 4) if best_combo_alphas else 0,
                    "avg_monthly_alpha": round(float(np.mean(best_combo_alphas)), 4) if best_combo_alphas else 0,
                    "worst_monthly_alpha": round(float(np.min(best_combo_alphas)), 4) if best_combo_alphas else 0
                })
        combo_stats.sort(key=lambda r: (r["coverage_rate"], r["avg_monthly_alpha"]), reverse=True)

        return jsonify({
            "params": {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "opt_window_x": x_win,
                "hurdle_y_pa": round(y_pa * 12, 4),
                "cost_rate": cost_rate,
                "step": step,
                "max_rules": max_rules,
                "target_rules": target_rules,
                "grid": "9m/12m"
            },
            "summary": {
                "months": len(month_rows),
                "covered_by_any_rule": len(month_rows) - len(uncovered),
                "any_rule_coverage_rate": round((len(month_rows) - len(uncovered)) / len(month_rows), 4) if month_rows else 0,
                "uncovered_months": len(uncovered)
            },
            "single_rule_stats": single_rule_stats,
            "target_rule_details": target_rule_details,
            "best_combinations": combo_stats[:20],
            "uncovered_months": uncovered
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
