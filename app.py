import os
import math
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

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


def get_regime(use_mf):
    if use_mf:
        base = BASE_WEIGHTS_FULL.copy()
        assets = list(base.keys()) + ["cash"]
    else:
        base = {
            "equities": 0.55,
            "bonds": 0.10,
            "gold": 0.10,
            "commodities": 0.10
        }
        s = sum(base.values())
        base = {k: v / s for k, v in base.items()}
        assets = list(base.keys()) + ["cash"]

    return base, assets


def extract_close_series(df, ticker):
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", ticker) in df.columns:
            s = df[("Close", ticker)]
        else:
            close_block = df.xs("Close", axis=1, level=0, drop_level=False)
            if close_block.shape[1] == 0:
                return None
            s = close_block.iloc[:, 0]
    else:
        if "Close" in df.columns:
            s = df["Close"]
        else:
            s = df.iloc[:, 0]

    if isinstance(s, pd.DataFrame):
        if s.shape[1] == 0:
            return None
        s = s.iloc[:, 0]

    s = pd.to_numeric(s, errors="coerce").dropna()

    if s.empty:
        return None

    return s


def load_data(use_mf):
    base, assets = get_regime(use_mf)

    price_series = {}

    for asset in assets:
        ticker = DEFAULT_TICKERS[asset]
        try:
            df = yf.download(ticker, start=START_DATE, auto_adjust=True, progress=False)
            s = extract_close_series(df, ticker)
            if s is None or s.empty:
                continue
            price_series[asset] = s.resample("ME").last()
        except Exception:
            continue

    if not price_series:
        raise ValueError("Keine Kursdaten geladen")

    prices = pd.concat(price_series, axis=1).sort_index()

    returns = {}
    for asset in prices.columns:
        s = prices[asset]
        if asset == "cash":
            returns[asset] = (pd.to_numeric(s, errors="coerce") / 100.0) / 12.0
        else:
            returns[asset] = pd.to_numeric(s, errors="coerce").pct_change()

    rets = pd.DataFrame(returns).sort_index()

    return prices, rets


def find_start(prices, assets):
    for i in range(12, len(prices)):
        ok = True
        for asset in assets:
            if asset not in prices.columns:
                ok = False
                break
            window = prices[asset].iloc[i - 12:i + 1]
            if window.isna().any():
                ok = False
                break
        if ok:
            return i
    return None


def ret(prices, asset, i, n):
    if asset not in prices.columns or i < n:
        return np.nan
    p0 = prices[asset].iloc[i - n]
    p1 = prices[asset].iloc[i]
    if pd.isna(p0) or pd.isna(p1) or p0 == 0:
        return np.nan
    return p1 / p0 - 1


def is_active(asset, prices, i, prev_active):
    if asset not in prices.columns:
        return False

    window10 = prices[asset].iloc[i - 9:i + 1]
    if len(window10) < 10 or window10.isna().any():
        return False

    price = prices[asset].iloc[i]
    sma = window10.mean()

    base = price > sma

    if prev_active:
        return base

    r3 = ret(prices, asset, i, 3)
    r6 = ret(prices, asset, i, 6)

    if asset == "equities":
        return bool(base or (pd.notna(r3) and r3 > 0))

    if asset in ["gold", "bonds"]:
        conds = [
            bool(base),
            bool(pd.notna(r3) and r3 > 0),
            bool(pd.notna(r6) and r6 > 0)
        ]
        return sum(conds) >= 2

    if asset in ["commodities", "managed_futures"]:
        return bool(base and pd.notna(r3) and r3 > 0)

    return bool(base)


def build_weights(prices, i, prev_active, settings):
    use_mf = settings["use_mf"]
    base, assets = get_regime(use_mf)

    active = {asset: is_active(asset, prices, i, prev_active.get(asset, False)) for asset in base}
    actives = [asset for asset, v in active.items() if v]
    n = len(actives)

    cash = 0.03 if n >= 5 else 0.05 if n == 4 else 0.085 if n == 3 else 0.11 if n == 2 else 0.15 if n == 1 else 0.30

    w = {asset: 0.0 for asset in assets}
    w["cash"] = cash

    if n == 0:
        if "bonds" in w:
            w["bonds"] = 0.3
        if "gold" in w:
            w["gold"] = 0.2
        if sum(w.values()) < 1.0:
            w["cash"] += 1.0 - sum(w.values())
        return pd.Series(w), active

    budget = 1.0 - cash
    s = sum(base[a] for a in actives)

    for asset in actives:
        w[asset] = budget * base[asset] / s

    total = sum(w.values())
    if total < 1.0:
        w["cash"] += 1.0 - total

    return pd.Series(w), active


def run(prices, rets, settings):
    base, assets = get_regime(settings["use_mf"])

    required_assets = list(base.keys()) + ["cash"]
    start = find_start(prices, required_assets)

    if start is None:
        raise ValueError("Nicht genügend Historie für die gewählten Assets")

    nav = [1.0]
    dates = [prices.index[start]]

    prev_w = pd.Series(0.0, index=assets)
    prev_w["cash"] = 1.0

    prev_active = {}
    cost_rate = settings["cost"]

    for i in range(start, len(prices) - 1):
        w, active = build_weights(prices, i, prev_active, settings)

        if any(asset not in rets.columns for asset in assets):
            continue

        r = rets.iloc[i + 1][assets]

        if r.isna().any():
            continue

        turnover = float((w - prev_w).abs().sum())
        cost = cost_rate * turnover / 2.0
        port = float((w * r).sum()) - cost

        nav.append(nav[-1] * (1.0 + port))
        dates.append(prices.index[i + 1])

        prev_w = w
        prev_active = active

    if len(nav) < 2:
        raise ValueError("Backtest konnte nicht berechnet werden")

    return pd.Series(nav, index=dates)


@app.route("/api/backtest", methods=["POST"])
def backtest():
    try:
        data = request.json or {}

        settings = {
            "use_mf": bool(data.get("use_managed_futures", True)),
            "cost": float(data.get("transaction_cost_rate", 0.001))
        }

        prices, rets = load_data(settings["use_mf"])
        nav = run(prices, rets, settings)

        if "equities" not in rets.columns or "bonds" not in rets.columns:
            raise ValueError("Benchmark-Daten fehlen")

        bench_ret = (
            rets[["equities", "bonds"]]
            .dropna()
            .mul(pd.Series(BENCHMARK_WEIGHTS), axis=1)
            .sum(axis=1)
        )

        bench = (1.0 + bench_ret[bench_ret.index >= nav.index[0]]).cumprod()

        idx = nav.index.intersection(bench.index)

        if len(idx) == 0:
            raise ValueError("Keine gemeinsame Historie zwischen Portfolio und Benchmark")

        return jsonify({
            "chart": {
                "dates": [d.strftime("%Y-%m-%d") for d in idx],
                "portfolio": nav.loc[idx].tolist(),
                "benchmark": bench.loc[idx].tolist()
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
