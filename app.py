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
# KONFIG
# ------------------------------------------------------------
START_DATE = "2000-01-01"

DEFAULT_TICKERS = {
    "equities": "ACWI",
    "bonds": "IEF",
    "gold": "GLD",
    "commodities": "DBC",
    "managed_futures": "DBMF",
    "cash": "^IRX"
}

BASE_WEIGHTS_FULL = {
    "equities": 0.55,
    "bonds": 0.10,
    "gold": 0.10,
    "commodities": 0.10,
    "managed_futures": 0.15
}

BENCHMARK_WEIGHTS = {"equities": 0.70, "bonds": 0.30}


# ------------------------------------------------------------
# REGIME
# ------------------------------------------------------------
def get_regime(use_mf):
    if use_mf:
        base = BASE_WEIGHTS_FULL.copy()
        assets = list(base.keys()) + ["cash"]
        defensive = ["managed_futures", "bonds", "gold", "cash"]
    else:
        base = {
            "equities": 0.55,
            "bonds": 0.10,
            "gold": 0.10,
            "commodities": 0.10
        }
        s = sum(base.values())
        base = {k: v/s for k, v in base.items()}
        assets = list(base.keys()) + ["cash"]
        defensive = ["bonds", "gold", "cash"]

    return base, assets, defensive


# ------------------------------------------------------------
# DATEN
# ------------------------------------------------------------
def load_data(use_mf):
    base, assets, _ = get_regime(use_mf)

    prices = {}
    for a in assets:
        t = DEFAULT_TICKERS[a]
        df = yf.download(t, start=START_DATE, auto_adjust=True, progress=False)

        if df.empty:
            continue

        s = df["Close"] if "Close" in df else df.iloc[:, 0]
        prices[a] = s.resample("ME").last()

    prices = pd.concat(prices, axis=1)

    # RETURNS
    rets = {}

    for a in prices.columns:
        s = prices[a]

        if a == "cash":
            rets[a] = (s / 100) / 12
        else:
            rets[a] = s.pct_change()

    rets = pd.DataFrame(rets)

    return prices, rets


# ------------------------------------------------------------
# STARTPUNKT
# ------------------------------------------------------------
def find_start(prices, assets):
    for i in range(12, len(prices)):
        ok = True
        for a in assets:
            window = prices[a].iloc[i-12:i+1]
            if window.isna().any():
                ok = False
        if ok:
            return i
    return None


# ------------------------------------------------------------
# MOMENTUM
# ------------------------------------------------------------
def ret(prices, a, i, n):
    if i < n:
        return np.nan
    return prices[a].iloc[i] / prices[a].iloc[i-n] - 1


# ------------------------------------------------------------
# AKTIV
# ------------------------------------------------------------
def is_active(a, prices, i, prev_active):

    price = prices[a].iloc[i]
    sma = prices[a].iloc[i-9:i+1].mean()

    base = price > sma

    if prev_active:
        return base

    r3 = ret(prices, a, i, 3)
    r6 = ret(prices, a, i, 6)

    if a == "equities":
        return sum([base, r3 > 0]) >= 1

    if a in ["gold", "bonds"]:
        return sum([base, r3 > 0, r6 > 0]) >= 2

    if a in ["commodities", "managed_futures"]:
        return base and (r3 > 0)

    return base


# ------------------------------------------------------------
# GEWICHTE
# ------------------------------------------------------------
def build_weights(prices, i, prev_active, settings):

    use_mf = settings["use_mf"]
    base, assets, defensive = get_regime(use_mf)

    active = {a: is_active(a, prices, i, prev_active.get(a, False)) for a in base}

    actives = [a for a, v in active.items() if v]
    n = len(actives)

    cash = 0.03 if n >= 5 else 0.05 if n==4 else 0.085 if n==3 else 0.11 if n==2 else 0.15 if n==1 else 0.3

    w = {a: 0 for a in assets}
    w["cash"] = cash

    if n == 0:
        w["bonds"] = 0.3
        w["gold"] = 0.2
        return pd.Series(w), active

    budget = 1 - cash

    s = sum(base[a] for a in actives)
    for a in actives:
        w[a] = budget * base[a]/s

    return pd.Series(w), active


# ------------------------------------------------------------
# BACKTEST
# ------------------------------------------------------------
def run(prices, rets, settings):

    base, assets, _ = get_regime(settings["use_mf"])
    start = find_start(prices, list(base.keys()) + ["cash"])

    nav = [1]
    dates = [prices.index[start]]

    prev_w = pd.Series(0, index=assets)
    prev_w["cash"] = 1

    prev_active = {}

    cost_rate = settings["cost"]

    for i in range(start, len(prices)-1):

        w, active = build_weights(prices, i, prev_active, settings)

        r = rets.iloc[i+1][assets]

        if r.isna().any():
            continue

        turnover = (w - prev_w).abs().sum()
        cost = cost_rate * turnover / 2

        port = (w * r).sum() - cost

        nav.append(nav[-1]*(1+port))
        dates.append(prices.index[i+1])

        prev_w = w
        prev_active = active

    return pd.Series(nav, index=dates)


# ------------------------------------------------------------
# API
# ------------------------------------------------------------
@app.route("/api/backtest", methods=["POST"])
def backtest():

    data = request.json or {}

    settings = {
        "use_mf": data.get("use_managed_futures", True),
        "cost": float(data.get("transaction_cost_rate", 0.001))
    }

    prices, rets = load_data(settings["use_mf"])
    nav = run(prices, rets, settings)

    bench = (rets[["equities","bonds"]]
             .dropna()
             .mul(pd.Series(BENCHMARK_WEIGHTS), axis=1)
             .sum(axis=1))

    bench = (1+bench[bench.index >= nav.index[0]]).cumprod()

    idx = nav.index.intersection(bench.index)

    return jsonify({
        "chart": {
            "dates": [d.strftime("%Y-%m-%d") for d in idx],
            "portfolio": nav.loc[idx].tolist(),
            "benchmark": bench.loc[idx].tolist()
        }
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
