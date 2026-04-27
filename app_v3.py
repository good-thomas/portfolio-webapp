import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- KONFIGURATION ---
START_DATE = "2015-01-01"  # Für Defense-Historie (DFND) oft passender Start
DEFAULT_COST_RATE = 0.001   # 0,1% Transaktionskosten
BASE_DIR = Path(__file__).resolve().parent
SG_TREND_FILE = BASE_DIR / "data" / "SG_TRD_IDX.TXT"
SG_TREND_COLUMN = "managed_futures"
RISK_FREE_ASSET = "cash"

# 1. Assetklassen-Universum (Neutrale Basis)
NEUTRAL_WEIGHTS = {
    "equities": 0.50,
    "bonds": 0.20,
    "managed_futures": 0.10,
    "gold": 0.10,
    "bitcoin": 0.05,
    "cash": 0.05
}

DEFAULT_TICKERS = {
    "equities": "ACWI",        # MSCI World Basis
    "bonds": "IEF",            # 7-10y US Treasuries
    "gold": "GLD",             # Gold Spot
    "managed_futures": "DBMF", # Fallback, falls SG-Datei fehlt
    "bitcoin": "BTC-USD",      # Krypto
    "cash": "^IRX"             # 13 Week Treasury Bill
}

# 2. GICS Buckets für das Aktien-Alpha (15% vom Portfolio)
GICS_BUCKETS = {
    "energy": "WNRG.DE", "materials": "XMWS.DE", "defense": "DFND.L",
    "industrials_ex": "WIND.DE", "cons_disc": "SC0G.DE", "cons_staples": "WCSS.DE",
    "pharma_bio": "BIOT.DE", "hc_equip": "IHI", "banks": "EXV1.DE",
    "fin_serv": "WFIN.DE", "software": "IGPT.DE", "semis": "VVSM.DE",
    "comm_serv": "XWTS.DE", "utilities": "WUTI.DE", "real_estate": "DPRE.DE"
}

BENCHMARK_WEIGHTS = {"equities": 0.70, "bonds": 0.30}


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
    return None if s.empty else s


def load_sg_trend_series(path=SG_TREND_FILE):
    if not path.exists():
        return None

    sg = pd.read_csv(
        path,
        header=None,
        names=["date", "open", "high", "low", "close", "volume"]
    )
    sg["date"] = pd.to_datetime(sg["date"].astype(str), format="%Y%m%d", errors="coerce")
    sg["close"] = pd.to_numeric(sg["close"], errors="coerce")
    sg = sg.dropna(subset=["date", "close"]).set_index("date").sort_index()
    sg = sg[~sg.index.duplicated(keep="last")]

    if sg.empty:
        raise ValueError("SG Trend Datei enthält keine verwertbaren Daten")

    return sg["close"].rename(SG_TREND_COLUMN).resample("ME").last()


def load_asset_series(asset):
    if asset == "managed_futures":
        sg = load_sg_trend_series()
        if sg is not None and not sg.empty:
            return sg

    ticker = DEFAULT_TICKERS[asset]
    df = yf.download(ticker, start=START_DATE, auto_adjust=True, progress=False)
    s = extract_close_series(df, ticker)
    if s is None or s.empty:
        return None
    return s.resample("ME").last()


def load_data(include_bitcoin=True):
    assets = [a for a in NEUTRAL_WEIGHTS.keys() if include_bitcoin or a != "bitcoin"]
    yf_tickers = [DEFAULT_TICKERS[a] for a in assets if a != "managed_futures"] + list(GICS_BUCKETS.values())

    data = yf.download(yf_tickers, start=START_DATE, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(yf_tickers[0])

    asset_series = {}
    for asset in assets:
        try:
            if asset == "managed_futures":
                s = load_asset_series(asset)
            else:
                s = data[DEFAULT_TICKERS[asset]] if DEFAULT_TICKERS[asset] in data.columns else None
            if s is not None and not s.empty:
                asset_series[asset] = pd.to_numeric(s, errors="coerce").resample("ME").last()
        except Exception:
            continue

    bucket_series = {}
    for bucket, ticker in GICS_BUCKETS.items():
        try:
            if ticker in data.columns:
                bucket_series[bucket] = pd.to_numeric(data[ticker], errors="coerce").resample("ME").last()
        except Exception:
            continue

    if not asset_series:
        raise ValueError("Keine Kursdaten geladen")

    prices = pd.concat({**asset_series, **bucket_series}, axis=1).sort_index().ffill()
    prices = prices[prices.index >= pd.to_datetime(START_DATE)]

    returns = {}
    for col in prices.columns:
        if col == RISK_FREE_ASSET:
            returns[col] = (pd.to_numeric(prices[col], errors="coerce") / 100.0) / 12.0
        else:
            returns[col] = pd.to_numeric(prices[col], errors="coerce").pct_change()

    rets = pd.DataFrame(returns).sort_index()
    return prices, rets


def compute_score(prices, asset, i):
    if asset not in prices.columns or i < 12:
        return np.nan
    p = prices[asset]
    vals = [p.iloc[i], p.iloc[i - 3], p.iloc[i - 6], p.iloc[i - 12]]
    if any(pd.isna(v) or v == 0 for v in vals[1:]) or pd.isna(vals[0]):
        return np.nan
    r3 = vals[0] / vals[1] - 1
    r6 = vals[0] / vals[2] - 1
    r12 = vals[0] / vals[3] - 1
    return (r3 * 0.5) + (r6 * 0.3) + (r12 * 0.2)


def has_required_history(prices, assets, i, lookback=12):
    if i < lookback:
        return False
    for asset in assets:
        if asset not in prices.columns:
            return False
        window = prices[asset].iloc[i - lookback:i + 1]
        if window.isna().any():
            return False
    return True


def resolve_start_index(prices, assets, start_date=None, lookback=12):
    candidate_positions = range(lookback, len(prices))
    if start_date:
        target_date = pd.to_datetime(start_date, errors="coerce")
        if pd.isna(target_date):
            raise ValueError("Ungültiges Startdatum. Format bitte YYYY-MM-DD")
        candidate_positions = np.where(prices.index >= target_date)[0]
        if len(candidate_positions) == 0:
            raise ValueError("Startdatum liegt nach der verfügbaren Historie")

    for i in candidate_positions:
        if has_required_history(prices, assets, int(i), lookback=lookback):
            return int(i)
    raise ValueError("Nicht genügend Historie für die gewählten Assets")


def build_v3_weights(prices, i, include_bitcoin=True):
    assets = [a for a in NEUTRAL_WEIGHTS.keys() if include_bitcoin or a != "bitcoin"]
    weights_base = {a: NEUTRAL_WEIGHTS[a] for a in assets}
    total_base = sum(weights_base.values())
    weights_base = {a: v / total_base for a, v in weights_base.items()}

    asset_scores = {
        asset: compute_score(prices, asset, i)
        for asset in assets
        if asset != RISK_FREE_ASSET
    }
    ranking = sorted(
        asset_scores,
        key=lambda a: asset_scores[a] if pd.notna(asset_scores[a]) else -999,
        reverse=True
    )
    top3 = ranking[:3]

    w_step = weights_base.copy()
    boosts = {0: 0.10, 1: 0.05, 2: 0.02}
    total_boost = 0.0

    for rank, asset in enumerate(top3):
        boost = boosts[rank]
        if pd.notna(asset_scores[asset]) and asset_scores[asset] > 0:
            w_step[asset] += boost
        else:
            w_step[RISK_FREE_ASSET] = w_step.get(RISK_FREE_ASSET, 0.0) + boost
        total_boost += boost

    if "equities" in w_step:
        w_step["equities"] -= total_boost * 0.7
    if "bonds" in w_step:
        w_step["bonds"] -= total_boost * 0.3

    bucket_scores = {
        bucket: compute_score(prices, bucket, i)
        for bucket in GICS_BUCKETS.keys()
        if bucket in prices.columns
    }
    top_buckets = sorted(
        bucket_scores,
        key=lambda b: bucket_scores[b] if pd.notna(bucket_scores[b]) else -999,
        reverse=True
    )[:3]

    all_cols = list(prices.columns)
    current_weights = pd.Series(0.0, index=all_cols)

    equity_alpha = min(0.15, max(0.0, w_step.get("equities", 0.0)))
    for asset, weight in w_step.items():
        if asset == "equities":
            current_weights["equities"] = max(0.0, weight - equity_alpha)
            if top_buckets and equity_alpha > 0:
                for tb in top_buckets:
                    current_weights[tb] = equity_alpha / len(top_buckets)
        else:
            current_weights[asset] = max(0.0, weight)

    weight_sum = current_weights.sum()
    if weight_sum > 0:
        current_weights = current_weights / weight_sum

    diagnostics = {
        "asset_scores": {k: None if pd.isna(v) else float(v) for k, v in asset_scores.items()},
        "asset_ranking": ranking,
        "selected_assets": top3,
        "bucket_scores": {k: None if pd.isna(v) else float(v) for k, v in bucket_scores.items()},
        "selected_buckets": top_buckets,
        "weighting_method": "neutral_base_plus_top3_momentum_boosts_and_top3_sector_alpha"
    }
    return current_weights, diagnostics


def run(prices, rets, settings):
    include_bitcoin = settings["include_bitcoin"]
    base_assets = [a for a in NEUTRAL_WEIGHTS.keys() if include_bitcoin or a != "bitcoin"]
    required_assets = base_assets + ["equities", "bonds"]
    start = resolve_start_index(prices, required_assets, settings.get("start_date"), lookback=12)

    nav = [1.0]
    dates = [prices.index[start]]
    prev_w = pd.Series(0.0, index=prices.columns)
    prev_w[RISK_FREE_ASSET] = 1.0 if RISK_FREE_ASSET in prev_w.index else 0.0
    history = []

    for i in range(start, len(prices) - 1):
        w, diagnostics = build_v3_weights(prices, i, include_bitcoin=include_bitcoin)
        r = rets.reindex(columns=w.index).iloc[i + 1]

        tradable = r.notna()
        w = w.where(tradable, 0.0)
        if w.sum() <= 0:
            continue
        w = w / w.sum()

        turnover = float((w - prev_w.reindex(w.index).fillna(0.0)).abs().sum())
        cost = settings["cost"] * turnover / 2.0
        gross_port = float((w * r.fillna(0.0)).sum())
        net_port = gross_port - cost

        nav.append(nav[-1] * (1.0 + net_port))
        dates.append(prices.index[i + 1])

        history.append({
            "date": prices.index[i + 1].strftime("%Y-%m-%d"),
            "weights": {k: float(v) for k, v in w.items() if abs(float(v)) > 1e-10},
            "turnover": turnover,
            "rebalancing_cost": cost,
            "gross_return": gross_port,
            "net_return": net_port,
            "v3": diagnostics
        })
        prev_w = w

    if len(nav) < 2:
        raise ValueError("Backtest konnte nicht berechnet werden")

    return pd.Series(nav, index=dates), history


def get_stats(nav, label):
    nav = nav.dropna().astype(float)
    if len(nav) < 2:
        return {"strategy": label, "cagr": "0.0 %", "vola": "0.0 %", "max_dd": "0.0 %", "sharpe": 0}

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


@app.route("/api/backtest_v3", methods=["POST"])
def backtest_v3():
    try:
        data = request.json or {}
        settings = {
            "include_bitcoin": bool(data.get("include_bitcoin", True)),
            "cost": float(data.get("transaction_cost_rate", DEFAULT_COST_RATE)),
            "start_date": data.get("start_date")
        }

        prices, rets = load_data(settings["include_bitcoin"])
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
        proxies = {asset: DEFAULT_TICKERS[asset] for asset in NEUTRAL_WEIGHTS if asset in DEFAULT_TICKERS}
        proxies["managed_futures"] = "data/SG_TRD_IDX.TXT" if SG_TREND_FILE.exists() else DEFAULT_TICKERS["managed_futures"]
        proxies.update({bucket: ticker for bucket, ticker in GICS_BUCKETS.items()})

        return jsonify({
            "summary": [
                get_stats(nav.loc[idx], "Portfolio V3"),
                get_stats(bench.loc[idx], "Benchmark 70/30")
            ],
            "chart": {
                "dates": [d.strftime("%Y-%m-%d") for d in idx],
                "portfolio": nav.loc[idx].tolist(),
                "benchmark": bench.loc[idx].tolist()
            },
            "latest_weights": latest,
            "history": history,
            "meta": {
                "include_bitcoin": settings["include_bitcoin"],
                "transaction_cost_rate": settings["cost"],
                "start_date": settings["start_date"],
                "actual_start_date": idx[0].strftime("%Y-%m-%d"),
                "ranking_method": "weighted_3m_6m_12m_momentum",
                "weighting_method": "neutral_base_plus_top3_asset_boosts_plus_top3_sector_alpha",
                "selected_asset_count": 3,
                "selected_sector_count": 3,
                "managed_futures_source": "local_sg_trend_file" if SG_TREND_FILE.exists() else "yfinance_dbmf",
                "proxies": proxies
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtest", methods=["POST"])
def backtest_alias():
    return backtest_v3()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
