import os
import math
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ------------------------------------------------------------
# KONFIGURATION
# ------------------------------------------------------------
START_DATE = "2000-01-01"

DEFAULT_TICKERS = {
    "equities": "ACWI",
    "bonds": "IEF",
    "gold": "GLD",
    "commodities": "DBC",
    "managed_futures": "DBMF",
    "cash": "SGOV"
}

BENCHMARK_WEIGHTS = {
    "equities": 0.70,
    "bonds": 0.30
}

BASE_WEIGHTS = {
    "equities": 0.55,
    "bonds": 0.10,
    "gold": 0.10,
    "commodities": 0.10,
    "managed_futures": 0.15
}

DEFENSIVE_PRIORITY = ["managed_futures", "bonds", "gold", "cash"]
RISK_ASSETS = ["equities", "commodities"]
ALL_ASSETS = ["equities", "bonds", "gold", "commodities", "managed_futures", "cash"]


# ------------------------------------------------------------
# DATENLADEN
# ------------------------------------------------------------
def download_close_series(ticker, start=START_DATE):
    data = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if data.empty:
        return pd.Series(dtype=float)

    if isinstance(data.columns, pd.MultiIndex):
        if ("Close", ticker) in data.columns:
            s = data[("Close", ticker)]
        else:
            s = data.xs("Close", axis=1, level=0).iloc[:, 0]
    else:
        s = data["Close"]

    return s.dropna().astype(float)


def load_monthly_data():
    prices = {}
    for asset, ticker in DEFAULT_TICKERS.items():
        s = download_close_series(ticker)
        if s.empty:
            continue
        prices[asset] = s.resample("ME").last()

    df_prices = pd.concat(prices, axis=1)
    df_prices.columns = list(prices.keys())
    df_prices = df_prices.sort_index().dropna(how="all").ffill()

    df_returns = df_prices.pct_change().fillna(0.0)

    return df_prices, df_returns


# ------------------------------------------------------------
# HILFSFUNKTIONEN
# ------------------------------------------------------------
def descending_rank(series_dict):
    s = pd.Series(series_dict, dtype=float)
    return s.rank(ascending=False, method="min")


def safe_return(prices, asset, idx, lookback):
    if idx < lookback:
        return np.nan
    p0 = prices[asset].iloc[idx - lookback]
    p1 = prices[asset].iloc[idx]
    if pd.isna(p0) or pd.isna(p1) or p0 == 0:
        return np.nan
    return p1 / p0 - 1


def get_10m_sma(prices, asset, idx):
    if idx < 9:
        return np.nan
    return prices[asset].iloc[idx - 9: idx + 1].mean()


def compute_cross_asset_momentum_rank(prices, idx):
    values = {}
    for asset in BASE_WEIGHTS.keys():
        if asset not in prices.columns:
            continue
        r3 = safe_return(prices, asset, idx, 3)
        r6 = safe_return(prices, asset, idx, 6)
        r12 = safe_return(prices, asset, idx, 12)
        vals = [x for x in [r3, r6, r12] if not pd.isna(x)]
        if len(vals) == 0:
            continue
        values[asset] = np.mean(vals)

    if not values:
        return pd.Series(dtype=float)

    return descending_rank(values)


# ------------------------------------------------------------
# AKTIV-/RE-ENTRY-REGELN
# ------------------------------------------------------------
def is_asset_active(asset, prices, idx, prev_active_map):
    if asset not in prices.columns:
        return False

    price = prices[asset].iloc[idx]
    sma10 = get_10m_sma(prices, asset, idx)

    if pd.isna(price) or pd.isna(sma10):
        return False

    base_rule = price > sma10
    was_prev_active = prev_active_map.get(asset, False)

    # Normale Regel für Assets, die bereits aktiv waren
    if was_prev_active:
        return base_rule

    # Schnellere Re-Entry-Regeln für vorher inaktive Assets
    r3 = safe_return(prices, asset, idx, 3)
    r6 = safe_return(prices, asset, idx, 6)

    if asset == "equities":
        momentum_ranks = compute_cross_asset_momentum_rank(prices, idx)
        rank_ok = asset in momentum_ranks.index and momentum_ranks[asset] <= 2
        conditions = [
            base_rule,
            (not pd.isna(r3) and r3 > 0),
            rank_ok
        ]
        return sum(bool(x) for x in conditions) >= 2

    if asset in ["gold", "bonds"]:
        conditions = [
            base_rule,
            (not pd.isna(r3) and r3 > 0),
            (not pd.isna(r6) and r6 > 0)
        ]
        return sum(bool(x) for x in conditions) >= 2

    if asset in ["commodities", "managed_futures"]:
        return base_rule and (not pd.isna(r3) and r3 > 0)

    return base_rule


# ------------------------------------------------------------
# TILT-SCORE
# ------------------------------------------------------------
def compute_tilt_scores(prices, idx, active_assets):
    if len(active_assets) == 0:
        return pd.Series(dtype=float)

    ret3 = {}
    ret6 = {}
    ret12 = {}

    for asset in active_assets:
        r3 = safe_return(prices, asset, idx, 3)
        r6 = safe_return(prices, asset, idx, 6)
        r12 = safe_return(prices, asset, idx, 12)

        if not pd.isna(r3):
            ret3[asset] = r3
        if not pd.isna(r6):
            ret6[asset] = r6
        if not pd.isna(r12):
            ret12[asset] = r12

    score = pd.Series(0.0, index=active_assets, dtype=float)

    # Höhere Momentum-Ränge -> höherer Score
    if ret3:
        ranks3 = descending_rank(ret3)
        score = score.add((len(ranks3) + 1 - ranks3), fill_value=0.0)

    if ret6:
        ranks6 = descending_rank(ret6)
        score = score.add((len(ranks6) + 1 - ranks6), fill_value=0.0)

    if ret12:
        ranks12 = descending_rank(ret12)
        score = score.add((len(ranks12) + 1 - ranks12), fill_value=0.0)

    return score.sort_values(ascending=False)


# ------------------------------------------------------------
# CASH-REGEL
# ------------------------------------------------------------
def determine_cash_weight(n_active):
    if n_active >= 5:
        return 0.03
    if n_active == 4:
        return 0.05
    if n_active == 3:
        return 0.085
    if n_active == 2:
        return 0.11
    if n_active == 1:
        return 0.15
    return 0.30


# ------------------------------------------------------------
# DEFENSIVE HIERARCHIE
# ------------------------------------------------------------
def distribute_inactive_risk_gaps(weights, active_map):
    # Nur echte Risk-Blöcke werden ausdrücklich defensiv ersetzt
    for asset in RISK_ASSETS:
        if active_map.get(asset, False):
            continue

        gap = BASE_WEIGHTS.get(asset, 0.0)
        if gap <= 0:
            continue

        for target in DEFENSIVE_PRIORITY:
            if target == "cash":
                weights["cash"] += gap
                break

            if weights.get(target, 0.0) > 0:
                weights[target] += gap
                break

    return weights


# ------------------------------------------------------------
# GEWICHTUNG
# ------------------------------------------------------------
def build_target_weights(prices, idx, settings, prev_active_map):
    active_map = {}
    for asset in BASE_WEIGHTS.keys():
        active_map[asset] = is_asset_active(asset, prices, idx, prev_active_map)

    active_assets = [a for a, active in active_map.items() if active]
    n_active = len(active_assets)

    cash_weight = determine_cash_weight(n_active)

    weights = {asset: 0.0 for asset in ALL_ASSETS}
    weights["cash"] = cash_weight

    # Spezialfall: nichts aktiv
    if n_active == 0:
        if "managed_futures" in DEFAULT_TICKERS:
            weights["managed_futures"] = 0.40
        if "bonds" in DEFAULT_TICKERS:
            weights["bonds"] = 0.20
        if "gold" in DEFAULT_TICKERS:
            weights["gold"] = 0.10

        allocated = sum(weights.values())
        if allocated < 1.0:
            weights["cash"] += 1.0 - allocated

        return pd.Series(weights), active_map, pd.Series(dtype=float)

    investable_budget = 1.0 - cash_weight

    # Kern proportional auf aktive Assets umlegen
    active_base_sum = sum(BASE_WEIGHTS[a] for a in active_assets)
    for asset in active_assets:
        weights[asset] = investable_budget * BASE_WEIGHTS[asset] / active_base_sum

    # Defensiv-Hierarchie für weggefallene Risk-Blöcke
    weights = distribute_inactive_risk_gaps(weights, active_map)

    # Moderater Tilt
    tilt_scores = compute_tilt_scores(prices, idx, active_assets)
    if not tilt_scores.empty and len(active_assets) > 1:
        base_active_weights = pd.Series({a: weights[a] for a in active_assets}, dtype=float)

        centered = tilt_scores - tilt_scores.mean()
        if centered.abs().sum() > 0:
            tilt_strength = float(settings.get("tilt_strength", 0.12))
            tilt_vector = centered / centered.abs().sum()

            adjusted = base_active_weights * (1.0 + tilt_strength * tilt_vector)
            adjusted = adjusted.clip(lower=0.0)

            if adjusted.sum() > 0:
                adjusted = adjusted / adjusted.sum() * base_active_weights.sum()
                for asset in active_assets:
                    weights[asset] = float(adjusted[asset])

    total = sum(weights.values())
    if abs(total - 1.0) > 1e-9:
        weights["cash"] += 1.0 - total

    return pd.Series(weights).reindex(ALL_ASSETS).fillna(0.0), active_map, tilt_scores


# ------------------------------------------------------------
# BACKTEST
# ------------------------------------------------------------
def run_strategy(prices, returns, settings):
    start_idx = 12
    dates = prices.index

    nav = [1.0]
    nav_dates = [dates[start_idx]]
    weight_records = []

    prev_active_map = {asset: False for asset in BASE_WEIGHTS.keys()}

    for idx in range(start_idx, len(dates) - 1):
        weights, active_map, tilt_scores = build_target_weights(prices, idx, settings, prev_active_map)

        next_ret = returns.iloc[idx + 1].reindex(ALL_ASSETS).fillna(0.0)
        port_ret = float((weights * next_ret).sum())

        nav.append(nav[-1] * (1.0 + port_ret))
        nav_dates.append(dates[idx + 1])

        weight_records.append({
            "date": dates[idx + 1].strftime("%Y-%m-%d"),
            **{k: float(v) for k, v in weights.items()},
            "active_assets": [a for a, flag in active_map.items() if flag],
            "tilt_scores": {k: float(v) for k, v in tilt_scores.to_dict().items()}
        })

        prev_active_map = active_map.copy()

    return pd.Series(nav, index=nav_dates, dtype=float), weight_records


def build_benchmark_nav(returns, start_date):
    bench_ret = returns[["equities", "bonds"]].mul(pd.Series(BENCHMARK_WEIGHTS), axis=1).sum(axis=1)
    bench_ret = bench_ret[bench_ret.index >= start_date]
    return (1.0 + bench_ret).cumprod()


# ------------------------------------------------------------
# STATS
# ------------------------------------------------------------
def get_stats(nav_series, label):
    nav_series = nav_series.dropna().astype(float)

    if len(nav_series) < 2:
        return {
            "strategy": label,
            "cagr": "0.0 %",
            "vola": "0.0 %",
            "max_dd": "0.0 %",
            "sharpe": 0
        }

    rets = nav_series.pct_change().dropna()

    start_val = float(nav_series.iloc[0])
    end_val = float(nav_series.iloc[-1])
    years = (nav_series.index[-1] - nav_series.index[0]).days / 365.25

    cagr_val = (end_val / start_val) ** (1 / years) - 1 if years > 0 and start_val > 0 else 0.0
    ann_vola = rets.std() * math.sqrt(12) if not rets.empty else 0.0
    drawdown = (nav_series / nav_series.cummax() - 1).min()
    sharpe = (rets.mean() / rets.std()) * math.sqrt(12) if not rets.empty and rets.std() != 0 else 0.0

    return {
        "strategy": label,
        "cagr": f"{cagr_val * 100:.1f} %",
        "vola": f"{ann_vola * 100:.1f} %",
        "max_dd": f"{drawdown * 100:.1f} %",
        "sharpe": round(float(sharpe), 2)
    }


# ------------------------------------------------------------
# API
# ------------------------------------------------------------
@app.route("/api/backtest", methods=["POST"])
def backtest():
    try:
        data = request.json or {}

        settings = {
            "tilt_strength": float(data.get("tilt_strength", 0.12))
        }

        prices, returns = load_monthly_data()
        portfolio_nav, weights_history = run_strategy(prices, returns, settings)
        benchmark_nav = build_benchmark_nav(returns, portfolio_nav.index[0])

        aligned_index = portfolio_nav.index.intersection(benchmark_nav.index)
        portfolio_nav = portfolio_nav.loc[aligned_index]
        benchmark_nav = benchmark_nav.loc[aligned_index]

        summary = [
            get_stats(portfolio_nav, "Portfolio"),
            get_stats(benchmark_nav, "Benchmark 70/30")
        ]

        return jsonify({
            "summary": summary,
            "chart": {
                "dates": [d.strftime("%Y-%m-%d") for d in aligned_index],
                "portfolio": portfolio_nav.tolist(),
                "benchmark": benchmark_nav.tolist()
            },
            "weights": weights_history,
            "meta": {
                "rebalance_frequency": "monthly",
                "benchmark": {"equities": "ACWI", "bonds": "IEF"},
                "proxies": DEFAULT_TICKERS,
                "note": "DBC enthält auch Gold-Exposure und ist daher nicht vollständig ex Gold."
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
