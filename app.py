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
RISK_FREE_ASSET = "cash"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SG_TREND_FILE = os.path.join(DATA_DIR, "SG_TRD_IDX.TXT")

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
        assets = list(base.keys()) + [RISK_FREE_ASSET]
    else:
        base = {
            "equities": 0.55,
            "bonds": 0.10,
            "gold": 0.10,
            "commodities": 0.10
        }
        s = sum(base.values())
        base = {k: v / s for k, v in base.items()}
        assets = list(base.keys()) + [RISK_FREE_ASSET]

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


def load_sg_trend_series(filepath=SG_TREND_FILE):
    if not os.path.exists(filepath):
        return None

    df = pd.read_csv(filepath, header=None)
    if df.shape[1] < 2:
        raise ValueError("SG Trend Datei hat zu wenige Spalten")

    df = df.iloc[:, :2].copy()
    df.columns = ["date", "value"]

    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["date", "value"]).sort_values("date")
    if df.empty:
        raise ValueError("SG Trend Datei enthält keine verwertbaren Daten")

    s = pd.Series(df["value"].values, index=df["date"], name="managed_futures")
    s = s[~s.index.duplicated(keep="last")]

    if s.empty:
        raise ValueError("SG Trend Serie ist leer")

    return s


def load_asset_series(asset):
    if asset == "managed_futures":
        sg_series = load_sg_trend_series()
        if sg_series is not None and not sg_series.empty:
            return sg_series.resample("ME").last()

    ticker = DEFAULT_TICKERS[asset]
    df = yf.download(ticker, start=START_DATE, auto_adjust=True, progress=False)
    s = extract_close_series(df, ticker)
    if s is None or s.empty:
        return None
    return s.resample("ME").last()


def load_data(use_mf):
    base, assets = get_regime(use_mf)

    price_series = {}

    for asset in assets:
        try:
            s = load_asset_series(asset)
            if s is None or s.empty:
                continue
            price_series[asset] = s
        except Exception:
            continue

    if not price_series:
        raise ValueError("Keine Kursdaten geladen")

    prices = pd.concat(price_series, axis=1).sort_index()

    returns = {}
    for asset in prices.columns:
        s = prices[asset]
        if asset == RISK_FREE_ASSET:
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


def get_momentum_ranks(prices, i, assets):
    moms = {}
    for asset in assets:
        if asset == RISK_FREE_ASSET:
            continue
        r3 = ret(prices, asset, i, 3)
        r6 = ret(prices, asset, i, 6)
        r12 = ret(prices, asset, i, 12)
        moms[asset] = np.nanmean([r3, r6, r12])

    series = pd.Series(moms).sort_values(ascending=False)
    ranks = {asset: rank + 1 for rank, asset in enumerate(series.index)}
    return ranks


def is_active_updated(asset, prices, i, prev_active, all_assets):
    if asset not in prices.columns:
        return False

    window10 = prices[asset].iloc[i - 9:i + 1]
    if len(window10) < 10 or window10.isna().any():
        return False
    base_signal = prices[asset].iloc[i] > window10.mean()

    if prev_active:
        return bool(base_signal)

    r3 = ret(prices, asset, i, 3)
    r6 = ret(prices, asset, i, 6)

    if asset == "equities":
        ranks = get_momentum_ranks(prices, i, all_assets)
        conds = [base_signal, (pd.notna(r3) and r3 > 0), (ranks.get(asset, 99) <= 2)]
        return sum(conds) >= 2

    if asset in ["gold", "bonds"]:
        conds = [base_signal, (pd.notna(r3) and r3 > 0), (pd.notna(r6) and r6 > 0)]
        return sum(conds) >= 2

    if asset in ["commodities", "managed_futures"]:
        return bool(base_signal and pd.notna(r3) and r3 > 0)

    return bool(base_signal)


def build_weights_with_tilt(prices, i, prev_active, settings):
    use_mf = settings["use_mf"]
    base_config, all_assets = get_regime(use_mf)

    active = {
        a: is_active_updated(a, prices, i, prev_active.get(a, False), list(base_config.keys()))
        for a in base_config
    }
    actives = [a for a, v in active.items() if v]
    n = len(actives)

    cash_map = {5: 0.03, 4: 0.05, 3: 0.08, 2: 0.11, 1: 0.15, 0: 0.35}
    cash_quote = cash_map.get(n, 0.35)

    w = {asset: 0.0 for asset in all_assets}

    if n == 0:
        remaining = 1.0 - 0.35
        w["managed_futures"] = remaining * 0.5 if "managed_futures" in w else 0.0
        w["bonds"] = remaining * 0.3
        w["gold"] = remaining * 0.2
        w[RISK_FREE_ASSET] = 1.0 - sum(w.values())
        return pd.Series(w), active

    budget = 1.0 - cash_quote
    raw_tilts = {a: max(0.1, 1 + ret(prices, a, i, 6)) for a in actives}

    total_weighted_base = sum(base_config[a] * raw_tilts[a] for a in actives)

    for a in actives:
        w[a] = budget * (base_config[a] * raw_tilts[a]) / total_weighted_base

    w[RISK_FREE_ASSET] = 1.0 - sum(w.values())
    return pd.Series(w), active


def run(prices, rets, settings):
    base, assets = get_regime(settings["use_mf"])

    required_assets = list(base.keys()) + [RISK_FREE_ASSET]
    start = find_start(prices, required_assets)

    if start is None:
        raise ValueError("Nicht genügend Historie für die gewählten Assets")

    nav = [1.0]
    dates = [prices.index[start]]

    prev_w = pd.Series(0.0, index=assets)
    prev_w[RISK_FREE_ASSET] = 1.0

    prev_active = {}
    cost_rate = settings["cost"]
    history = []

    for i in range(start, len(prices) - 1):
        w, active = build_weights_with_tilt(prices, i, prev_active, settings)

        if any(asset not in rets.columns for asset in assets):
            continue

        r = rets.iloc[i + 1][assets]

        if r.isna().any():
            continue

        turnover = float((w - prev_w).abs().sum())
        cost = cost_rate * turnover / 2.0
        gross_port = float((w * r).sum())
        net_port = gross_port - cost

        nav.append(nav[-1] * (1.0 + net_port))
        dates.append(prices.index[i + 1])

        history.append({
            "date": prices.index[i + 1].strftime("%Y-%m-%d"),
            "weights": {k: float(v) for k, v in w.items()},
            "turnover": turnover,
            "rebalancing_cost": cost,
            "gross_return": gross_port,
            "net_return": net_port,
            "active_assets": [k for k, v in active.items() if v]
        })

        prev_w = w
        prev_active = active

    if len(nav) < 2:
        raise ValueError("Backtest konnte nicht berechnet werden")

    return pd.Series(nav, index=dates), history


def get_stats(nav, label):
    nav = nav.dropna().astype(float)

    if len(nav) < 2:
        return {
            "strategy": label,
            "cagr": "0.0 %",
            "vola": "0.0 %",
            "max_dd": "0.0 %",
            "sharpe": 0
        }

    rets = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25

    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0
    vola = rets.std() * math.sqrt(12) if len(rets) > 1 else 0.0
    dd = (nav / nav.cummax() - 1).min()
    sharpe = (rets.mean() / rets.std()) * math.sqrt(12) if len(rets) > 1 and rets.std() != 0 else 0.0

    return {
        "strategy": label,
        "cagr": f"{cagr * 100:.1f} %",
        "vola": f"{vola * 100:.1f} %",
        "max_dd": f"{dd * 100:.1f} %",
        "sharpe": round(float(sharpe), 2)
    }


@app.route("/api/backtest", methods=["POST"])
def backtest():
    try:
        data = request.json or {}

        settings = {
            "use_mf": bool(data.get("use_managed_futures", True)),
            "cost": float(data.get("transaction_cost_rate", 0.001))
        }

        prices, rets = load_data(settings["use_mf"])
        nav, history = run(prices, rets, settings)

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

        latest = history[-1] if history else None
        proxies = {
            asset: DEFAULT_TICKERS[asset]
            for asset in get_regime(settings["use_mf"])[1]
            if asset in DEFAULT_TICKERS
        }
        if settings["use_mf"]:
            proxies["managed_futures"] = "SG_TRD_IDX.TXT" if os.path.exists(SG_TREND_FILE) else DEFAULT_TICKERS["managed_futures"]

        return jsonify({
            "summary": [
                get_stats(nav.loc[idx], "Portfolio"),
                get_stats(bench.loc[idx], "Benchmark 70/30")
            ],
            "chart": {
                "dates": [d.strftime("%Y-%m-%d") for d in idx],
                "portfolio": nav.loc[idx].tolist(),
                "benchmark": bench.loc[idx].tolist()
            },
            "latest_weights": latest,
            "meta": {
                "use_managed_futures": settings["use_mf"],
                "transaction_cost_rate": settings["cost"],
                "managed_futures_source": "local_sg_trend_file" if os.path.exists(SG_TREND_FILE) else "yfinance_dbmf",
                "proxies": proxies
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
