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
ROLLING_VOL_MONTHS = 6
RISK_FREE_ASSET = "cash"

DEFAULT_TICKERS = {
    "equities": "ACWI",
    "bonds": "IEF",
    "gold": "GLD",
    "commodities": "DBC",
    "managed_futures": "DBMF",
    "bitcoin": "BTC-USD",
    "cash": "^IRX"
}

BENCHMARK_WEIGHTS = {"equities": 0.70, "bonds": 0.30}


def get_universe(use_mf=True, include_bitcoin=True):
    risky_assets = ["equities", "bonds", "gold", "commodities"]
    if use_mf:
        risky_assets.append("managed_futures")
    if include_bitcoin:
        risky_assets.append("bitcoin")
    all_assets = risky_assets + [RISK_FREE_ASSET]
    return risky_assets, all_assets


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


def load_data(use_mf=True, include_bitcoin=True):
    risky_assets, all_assets = get_universe(use_mf, include_bitcoin)
    price_series = {}

    for asset in all_assets:
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
        if asset == RISK_FREE_ASSET:
            returns[asset] = (pd.to_numeric(s, errors="coerce") / 100.0) / 12.0
        else:
            returns[asset] = pd.to_numeric(s, errors="coerce").pct_change()

    rets = pd.DataFrame(returns).sort_index()
    return prices, rets


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


def find_start(prices, assets, lookback=12):
    for i in range(lookback, len(prices)):
        if has_required_history(prices, assets, i, lookback=lookback):
            return i
    return None


def resolve_start_index(prices, assets, start_date=None, lookback=12):
    if not start_date:
        return find_start(prices, assets, lookback=lookback)

    try:
        target_date = pd.Timestamp(start_date)
    except Exception:
        raise ValueError("Ungültiges Startdatum. Format bitte YYYY-MM-DD")

    candidate_positions = np.where(prices.index >= target_date)[0]
    if len(candidate_positions) == 0:
        raise ValueError("Startdatum liegt nach der verfügbaren Historie")

    for i in candidate_positions:
        if has_required_history(prices, assets, i, lookback=lookback):
            return int(i)

    raise ValueError("Für das gewählte Startdatum ist nicht genug Historie vorhanden")


def ret(prices, asset, i, n):
    if asset not in prices.columns or i < n:
        return np.nan
    p0 = prices[asset].iloc[i - n]
    p1 = prices[asset].iloc[i]
    if pd.isna(p0) or pd.isna(p1) or p0 == 0:
        return np.nan
    return p1 / p0 - 1


def compute_asset_score(prices, rets, asset, i):
    r3 = ret(prices, asset, i, 3)
    r6 = ret(prices, asset, i, 6)
    r12 = ret(prices, asset, i, 12)

    momentum = 0.25 * r3 + 0.50 * r6 + 0.25 * r12

    if asset not in rets.columns or i < ROLLING_VOL_MONTHS:
        return np.nan

    vol_window = pd.to_numeric(rets[asset].iloc[i - ROLLING_VOL_MONTHS + 1:i + 1], errors="coerce").dropna()
    if len(vol_window) < ROLLING_VOL_MONTHS:
        return np.nan

    vol = vol_window.std()
    if pd.isna(vol) or vol <= 0:
        return np.nan

    return momentum / vol


def run_pairwise_ranking(prices, rets, i, risky_assets):
    scores = {asset: compute_asset_score(prices, rets, asset, i) for asset in risky_assets}
    wins = {asset: 0 for asset in risky_assets}

    for idx_a in range(len(risky_assets)):
        for idx_b in range(idx_a + 1, len(risky_assets)):
            a = risky_assets[idx_a]
            b = risky_assets[idx_b]
            sa = scores.get(a, np.nan)
            sb = scores.get(b, np.nan)

            if pd.isna(sa) and pd.isna(sb):
                continue
            if pd.isna(sa):
                wins[b] += 1
                continue
            if pd.isna(sb):
                wins[a] += 1
                continue

            if sa > sb:
                wins[a] += 1
            elif sb > sa:
                wins[b] += 1
            else:
                r12_a = ret(prices, a, i, 12)
                r12_b = ret(prices, b, i, 12)
                if pd.notna(r12_a) and pd.notna(r12_b) and r12_a != r12_b:
                    if r12_a > r12_b:
                        wins[a] += 1
                    else:
                        wins[b] += 1
                else:
                    vol_a = pd.to_numeric(rets[a].iloc[i - ROLLING_VOL_MONTHS + 1:i + 1], errors="coerce").dropna().std()
                    vol_b = pd.to_numeric(rets[b].iloc[i - ROLLING_VOL_MONTHS + 1:i + 1], errors="coerce").dropna().std()
                    if pd.notna(vol_a) and pd.notna(vol_b):
                        if vol_a < vol_b:
                            wins[a] += 1
                        elif vol_b < vol_a:
                            wins[b] += 1

    ranking = sorted(
        risky_assets,
        key=lambda asset: (
            wins.get(asset, 0),
            ret(prices, asset, i, 12) if pd.notna(ret(prices, asset, i, 12)) else -999,
            -(pd.to_numeric(rets[asset].iloc[i - ROLLING_VOL_MONTHS + 1:i + 1], errors="coerce").dropna().std()
              if asset in rets.columns and i >= ROLLING_VOL_MONTHS else 999)
        ),
        reverse=True
    )

    top3 = ranking[:3]
    return scores, wins, ranking, top3


def compute_cov_matrix(rets, assets, i, window=6):
    if i < window:
        return None
    frame = rets[assets].iloc[i - window + 1:i + 1].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < window:
        return None
    cov = frame.cov()
    if cov.isna().any().any():
        return None
    return cov


def compute_portfolio_vol(weights, cov):
    w = np.array(weights, dtype=float)
    cov_m = np.array(cov, dtype=float)
    var = float(w.T @ cov_m @ w)
    return math.sqrt(max(var, 0.0))


def build_risk_targeted_weights(prices, rets, i, risky_assets, risk_factor):
    scores, wins, ranking, top3 = run_pairwise_ranking(prices, rets, i, risky_assets)

    positive_scores = {a: max(float(scores[a]), 0.0) for a in top3 if pd.notna(scores.get(a))}
    if not positive_scores:
        positive_scores = {a: 1.0 for a in top3}

    score_sum = sum(positive_scores.values())
    raw_weights = {a: positive_scores[a] / score_sum for a in top3}

    bench_assets = ["equities", "bonds"]
    if any(a not in rets.columns for a in bench_assets):
        raise ValueError("Benchmark-Daten fehlen für Risiko-Targeting")

    bench_cov = compute_cov_matrix(rets, bench_assets, i, window=ROLLING_VOL_MONTHS)
    top_cov = compute_cov_matrix(rets, top3, i, window=ROLLING_VOL_MONTHS)

    if bench_cov is None or top_cov is None:
        raise ValueError("Nicht genügend Historie für Risiko-Targeting")

    bench_vector = [BENCHMARK_WEIGHTS[a] for a in bench_assets]
    benchmark_vol = compute_portfolio_vol(bench_vector, bench_cov)
    raw_vector = [raw_weights[a] for a in top3]
    raw_vol = compute_portfolio_vol(raw_vector, top_cov)

    if raw_vol <= 0:
        invest_scale = 0.0
    else:
        target_vol = benchmark_vol * float(risk_factor)
        invest_scale = min(1.0, target_vol / raw_vol)

    final_weights = {asset: 0.0 for asset in risky_assets + [RISK_FREE_ASSET]}
    for asset in top3:
        final_weights[asset] = raw_weights[asset] * invest_scale
    final_weights[RISK_FREE_ASSET] = max(0.0, 1.0 - sum(final_weights.values()))

    diagnostics = {
        "scores": {k: None if pd.isna(v) else float(v) for k, v in scores.items()},
        "wins": wins,
        "ranking": ranking,
        "selected_assets": top3,
        "raw_weights": raw_weights,
        "benchmark_vol_monthly": float(benchmark_vol),
        "raw_portfolio_vol_monthly": float(raw_vol),
        "target_vol_monthly": float(benchmark_vol * float(risk_factor)),
        "invest_scale": float(invest_scale)
    }

    return pd.Series(final_weights), diagnostics


def run(prices, rets, settings):
    risky_assets, all_assets = get_universe(settings["use_mf"], settings["include_bitcoin"])
    required_assets = list(risky_assets) + [RISK_FREE_ASSET, "equities", "bonds"]

    start = resolve_start_index(
        prices=prices,
        assets=required_assets,
        start_date=settings.get("start_date"),
        lookback=12
    )
    if start is None:
        raise ValueError("Nicht genügend Historie für die gewählten Assets")

    nav = [1.0]
    dates = [prices.index[start]]
    prev_w = pd.Series(0.0, index=all_assets)
    prev_w[RISK_FREE_ASSET] = 1.0
    cost_rate = settings["cost"]
    history = []

    for i in range(start, len(prices) - 1):
        if any(asset not in rets.columns for asset in all_assets):
            continue

        w, diagnostics = build_risk_targeted_weights(
            prices=prices,
            rets=rets,
            i=i,
            risky_assets=risky_assets,
            risk_factor=settings["risk_factor"]
        )

        r = rets.iloc[i + 1][all_assets]
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
            "pairwise": diagnostics
        })

        prev_w = w

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


@app.route("/api/backtest_v2", methods=["POST"])
def backtest_v2():
    try:
        data = request.json or {}

        settings = {
            "use_mf": bool(data.get("use_managed_futures", True)),
            "include_bitcoin": bool(data.get("include_bitcoin", True)),
            "cost": float(data.get("transaction_cost_rate", 0.001)),
            "risk_factor": float(data.get("risk_factor", 1.0)),
            "start_date": data.get("start_date")
        }

        prices, rets = load_data(settings["use_mf"], settings["include_bitcoin"])
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
        risky_assets, all_assets = get_universe(settings["use_mf"], settings["include_bitcoin"])

        return jsonify({
            "summary": [
                get_stats(nav.loc[idx], "Portfolio V2"),
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
                "use_managed_futures": settings["use_mf"],
                "include_bitcoin": settings["include_bitcoin"],
                "transaction_cost_rate": settings["cost"],
                "risk_factor": settings["risk_factor"],
                "start_date": settings["start_date"],
                "ranking_method": "pairwise_wins_on_weighted_momentum_div_vol",
                "weighting_method": "score_proportional_then_risk_targeted",
                "selected_count": 3,
                "proxies": {asset: DEFAULT_TICKERS[asset] for asset in all_assets}
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
