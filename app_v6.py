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

START_DATE = "2000-01-01"
LOOKBACK_MONTHS = 12
DEFAULT_COST_RATE = 0.001

DEFAULT_EQUITY_FACTOR_MAX = 0.50
MAX_EQUITY_FACTOR_MAX = 0.80
EQUITY_ENGINE_TOP_N = 3
EQUITY_ENGINE_EDGE_POWER = 1.5
EQUITY_ENGINE_EQUAL_WEIGHT_BLEND = 0.70
EQUITY_ENGINE_EDGE_WEIGHT_BLEND = 0.30

COMMODITY_ENGINE_ASSET = "commodity_engine"
COMMODITY_ENGINE_SCORE_POWER = 1.5
BROAD_COMMODITY_SCORE_BOOST = 1.10
MAX_TOTAL_COMMODITIES = 0.30
MAX_BROAD_COMMODITIES = 0.20
MAX_SINGLE_COMMODITY_SUB = 0.125

DEFAULT_SECTOR_PROXY_SET = "us_long_history"
ALLOWED_SECTOR_PROXY_SETS = {"ucits", "us_long_history"}
BASE_DIR = Path(__file__).resolve().parent
SG_TREND_FILE = BASE_DIR / "data" / "SG_TRD_IDX.TXT"
SG_TREND_COLUMN = "managed_futures"
RISK_FREE_ASSET = "cash"
EQUITY_ENGINE_ASSET = "equity_engine"
MIN_CORE_ASSETS = ["equities", "bonds", "managed_futures", "gold", "cash"]

BENCHMARK_RELATIVE_FILTER = True
BENCHMARK_FALLBACK_MODE = "benchmark"
EQUITY_ENGINE_MIN_TOP1 = 0.60
EQUITY_ENGINE_MIN_STRONG_TOP1 = 0.70
EQUITY_ENGINE_STRONG_EDGE = 0.05
EQUITY_ENGINE_MIN_TOP2 = 0.55

NEUTRAL_WEIGHTS = {
    EQUITY_ENGINE_ASSET: 0.45,
    "bonds": 0.17,
    "managed_futures": 0.10,
    "gold": 0.10,
    COMMODITY_ENGINE_ASSET: 0.08,
    "bitcoin": 0.05,
    "cash": 0.05,
}

DEFAULT_TICKERS = {
    "equities": "ACWI",
    "bonds": "IEF",
    "gold": "GLD",
    "managed_futures": "DBMF",
    "bitcoin": "BTC-USD",
    "cash": "^IRX",
}

COMMODITY_BUCKETS_US_LONG_HISTORY = {
    "broad_commodities": "DBC",
    "energy_commodities": "USO",
    "industrial_metals": "DBB",
    "agriculture": "DBA",
    "precious_metals_ex_gold": "SLV",
    "uranium": "URA",
}

GICS_BUCKETS_UCITS = {
    "energy": "WNRG.DE", "materials": "XMWS.DE", "defense": "DFND.L",
    "industrials_ex": "WIND.DE", "cons_disc": "SC0G.DE", "cons_staples": "WCSS.DE",
    "pharma_bio": "BIOT.DE", "hc_equip": "IHI", "banks": "EXV1.DE",
    "fin_serv": "WFIN.DE", "software": "IGPT.DE", "semis": "VVSM.DE",
    "comm_serv": "XWTS.DE", "utilities": "WUTI.DE", "real_estate": "DPRE.DE",
}

GICS_BUCKETS_US_LONG_HISTORY = {
    "energy": "XLE", "materials": "XLB", "industrials": "XLI",
    "cons_disc": "XLY", "cons_staples": "XLP", "health_care": "XLV",
    "financials": "XLF", "technology": "XLK", "utilities": "XLU",
    "real_estate": "IYR", "semis": "SMH", "biotech": "IBB",
    "hc_equip": "IHI", "banks": "KBE", "aerospace_defense": "ITA",
    "infrastructure": "PAVE", "homebuilders": "XHB", "transportation": "IYT",
    "gold_miners": "GDX", "oil_services": "OIH", "metals_mining": "XME",
    "software": "IGV",
}

SECTOR_PROXY_SETS = {
    "ucits": GICS_BUCKETS_UCITS,
    "us_long_history": GICS_BUCKETS_US_LONG_HISTORY,
}

BENCHMARK_WEIGHTS = {"equities": 0.70, "bonds": 0.30}


def get_sector_buckets(sector_proxy_set):
    if sector_proxy_set not in SECTOR_PROXY_SETS:
        raise ValueError("sector_proxy_set muss einer dieser Werte sein: " + ", ".join(sorted(ALLOWED_SECTOR_PROXY_SETS)))
    return SECTOR_PROXY_SETS[sector_proxy_set]


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
        s = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
    if isinstance(s, pd.DataFrame):
        if s.shape[1] == 0:
            return None
        s = s.iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce").dropna()
    return None if s.empty else s


def load_sg_trend_series(path=SG_TREND_FILE):
    if not path.exists():
        return None
    sg = pd.read_csv(path, header=None, names=["date", "open", "high", "low", "close", "volume"])
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


def load_data(include_bitcoin=True, sector_proxy_set=DEFAULT_SECTOR_PROXY_SET):
    gics_buckets = get_sector_buckets(sector_proxy_set)
    commodity_buckets = COMMODITY_BUCKETS_US_LONG_HISTORY
    assets = ["equities", "bonds", "gold", "managed_futures", "cash"]
    if include_bitcoin:
        assets.append("bitcoin")

    yf_tickers = (
        [DEFAULT_TICKERS[a] for a in assets if a != "managed_futures"]
        + list(gics_buckets.values())
        + list(commodity_buckets.values())
    )
    yf_tickers = sorted(set(yf_tickers))
    data = yf.download(yf_tickers, start=START_DATE, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(yf_tickers[0])

    series = {}
    for asset in assets:
        try:
            if asset == "managed_futures":
                s = load_asset_series(asset)
            else:
                ticker = DEFAULT_TICKERS[asset]
                s = data[ticker] if ticker in data.columns else None
            if s is not None and not s.empty:
                series[asset] = pd.to_numeric(s, errors="coerce").resample("ME").last()
        except Exception:
            continue

    for bucket, ticker in gics_buckets.items():
        try:
            if ticker in data.columns:
                series[bucket] = pd.to_numeric(data[ticker], errors="coerce").resample("ME").last()
        except Exception:
            continue

    for bucket, ticker in commodity_buckets.items():
        try:
            if ticker in data.columns:
                series[bucket] = pd.to_numeric(data[ticker], errors="coerce").resample("ME").last()
        except Exception:
            continue

    if not series:
        raise ValueError("Keine Kursdaten geladen")

    prices = pd.concat(series, axis=1).sort_index().ffill()
    prices = prices[prices.index >= pd.to_datetime(START_DATE)]

    returns = {}
    for col in prices.columns:
        if col == RISK_FREE_ASSET:
            returns[col] = (pd.to_numeric(prices[col], errors="coerce") / 100.0) / 12.0
        else:
            returns[col] = pd.to_numeric(prices[col], errors="coerce").pct_change()
    return prices, pd.DataFrame(returns).sort_index(), gics_buckets, commodity_buckets


def has_lookback(prices, asset, i, lookback=LOOKBACK_MONTHS):
    if asset not in prices.columns or i < lookback:
        return False
    window = prices[asset].iloc[i - lookback:i + 1]
    return not window.isna().any()


def first_available_use_date(prices, asset, lookback=LOOKBACK_MONTHS):
    for i in range(lookback, len(prices)):
        if has_lookback(prices, asset, i, lookback):
            return prices.index[i]
    return None


def compute_availability(prices, assets, buckets):
    out = {}
    for name in list(assets) + list(buckets):
        dt = first_available_use_date(prices, name)
        out[name] = None if dt is None else dt.strftime("%Y-%m-%d")
    return out


def resolve_start_index(prices, start_date=None):
    if start_date:
        target = pd.to_datetime(start_date, errors="coerce")
        if pd.isna(target):
            raise ValueError("Ungültiges Startdatum. Format bitte YYYY-MM-DD")
        candidate_positions = np.where(prices.index >= target)[0]
    else:
        candidate_positions = range(LOOKBACK_MONTHS, len(prices))

    for i in candidate_positions:
        if all(has_lookback(prices, asset, int(i)) for asset in MIN_CORE_ASSETS):
            return int(i)
    raise ValueError("Nicht genügend Historie für den Mindest-Core")


def compute_score(prices, asset, i):
    if not has_lookback(prices, asset, i):
        return np.nan
    p = prices[asset]
    p_now, p3, p6, p12 = p.iloc[i], p.iloc[i - 3], p.iloc[i - 6], p.iloc[i - 12]
    if any(pd.isna(v) or v == 0 for v in [p3, p6, p12]) or pd.isna(p_now):
        return np.nan
    return 0.5 * (p_now / p3 - 1) + 0.3 * (p_now / p6 - 1) + 0.2 * (p_now / p12 - 1)


def compute_benchmark_score(prices, i):
    equity_score = compute_score(prices, "equities", i)
    bond_score = compute_score(prices, "bonds", i)
    if pd.isna(equity_score) or pd.isna(bond_score):
        return np.nan
    return BENCHMARK_WEIGHTS["equities"] * equity_score + BENCHMARK_WEIGHTS["bonds"] * bond_score


def clamp_factor_max(value):
    if value is None:
        return DEFAULT_EQUITY_FACTOR_MAX
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError("equity_factor_max muss eine Zahl zwischen 0.0 und 0.8 sein")
    if not np.isfinite(value):
        raise ValueError("equity_factor_max muss eine endliche Zahl zwischen 0.0 und 0.8 sein")
    return min(MAX_EQUITY_FACTOR_MAX, max(0.0, value))


def resolve_sector_proxy_set(value):
    if value is None:
        return DEFAULT_SECTOR_PROXY_SET
    value = str(value).strip()
    if value not in ALLOWED_SECTOR_PROXY_SETS:
        raise ValueError("sector_proxy_set muss einer dieser Werte sein: " + ", ".join(sorted(ALLOWED_SECTOR_PROXY_SETS)))
    return value


def normalize_weights(weights):
    weights = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(weights.values())
    if total <= 0:
        return weights
    return {k: v / total for k, v in weights.items()}


def cap_and_redistribute(weights, caps, max_iter=20):
    weights = normalize_weights(weights)
    if not weights:
        return weights
    locked = set()
    for _ in range(max_iter):
        changed = False
        excess = 0.0
        free_total = 0.0
        for k, w in list(weights.items()):
            cap = caps.get(k)
            if cap is not None and w > cap:
                excess += w - cap
                weights[k] = cap
                locked.add(k)
                changed = True
        if not changed or excess <= 1e-12:
            break
        for k, w in weights.items():
            if k not in locked:
                free_total += w
        if free_total <= 0:
            break
        for k, w in list(weights.items()):
            if k not in locked:
                weights[k] = w + excess * (w / free_total)
    return normalize_weights(weights)


def compute_blended_weights(top_items, item_scores, benchmark_score, total_weight, power=EQUITY_ENGINE_EDGE_POWER):
    if not top_items or total_weight <= 0 or pd.isna(benchmark_score):
        return {}, {}, {}, {}, {}
    equal_weights = {b: total_weight / len(top_items) for b in top_items}
    edges = {b: max(0.0, float(item_scores[b] - benchmark_score)) for b in top_items if pd.notna(item_scores.get(b))}
    edge_power_scores = {b: edge ** power for b, edge in edges.items()}
    power_sum = sum(edge_power_scores.values())
    edge_weights = dict(equal_weights) if power_sum <= 0 else {b: total_weight * edge_power_scores[b] / power_sum for b in edge_power_scores}
    blended_weights = {}
    for b in top_items:
        blended_weights[b] = EQUITY_ENGINE_EQUAL_WEIGHT_BLEND * equal_weights.get(b, 0.0) + EQUITY_ENGINE_EDGE_WEIGHT_BLEND * edge_weights.get(b, 0.0)
    total = sum(blended_weights.values())
    if total > 0:
        blended_weights = {b: total_weight * w / total for b, w in blended_weights.items()}
    return blended_weights, equal_weights, edge_weights, edges, edge_power_scores


def compute_equity_engine_weights(prices, i, gics_buckets, factor_max=DEFAULT_EQUITY_FACTOR_MAX):
    equity_score = compute_score(prices, "equities", i)
    available_buckets = [b for b in gics_buckets if has_lookback(prices, b, i)]
    bucket_scores = {b: compute_score(prices, b, i) for b in available_buckets}
    candidates = [b for b in available_buckets if pd.notna(bucket_scores.get(b)) and pd.notna(equity_score) and bucket_scores[b] > equity_score and bucket_scores[b] > 0]
    top_buckets = sorted(candidates, key=lambda b: bucket_scores[b], reverse=True)[:EQUITY_ENGINE_TOP_N]

    if not top_buckets or pd.isna(equity_score):
        diagnostics = {
            "equity_score": None if pd.isna(equity_score) else float(equity_score),
            "available_buckets": available_buckets,
            "bucket_scores": {k: None if pd.isna(v) else float(v) for k, v in bucket_scores.items()},
            "relative_strength_buckets": candidates,
            "selected_buckets": [],
            "factor_weight": 0.0,
            "base_weight": 1.0,
            "internal_weights": {"equities": 1.0},
        }
        return {"equities": 1.0}, diagnostics

    relative_edges = [bucket_scores[b] - equity_score for b in top_buckets]
    avg_relative_edge = float(np.mean(relative_edges))
    max_relative_edge = float(np.max(relative_edges))
    scale = 0.0 if avg_relative_edge <= 0 else (0.66 if avg_relative_edge < 0.02 else 1.0)
    factor_weight = float(factor_max * scale)
    base_weight = float(1.0 - factor_weight)
    sector_weights, equal_weights, edge_weights, edges, edge_power_scores = compute_blended_weights(top_buckets, bucket_scores, equity_score, factor_weight)
    internal_weights = normalize_weights({"equities": base_weight, **sector_weights})
    diagnostics = {
        "equity_score": float(equity_score),
        "available_buckets": available_buckets,
        "bucket_scores": {k: None if pd.isna(v) else float(v) for k, v in bucket_scores.items()},
        "relative_strength_buckets": candidates,
        "selected_buckets": top_buckets,
        "factor_max": float(factor_max),
        "factor_weight": float(factor_weight),
        "base_weight": float(base_weight),
        "avg_relative_edge": avg_relative_edge,
        "max_relative_edge": max_relative_edge,
        "equal_weights": {k: float(v) for k, v in equal_weights.items()},
        "edge_weights": {k: float(v) for k, v in edge_weights.items()},
        "edges": {k: float(v) for k, v in edges.items()},
        "edge_power_scores": {k: float(v) for k, v in edge_power_scores.items()},
        "internal_weights": {k: float(v) for k, v in internal_weights.items()},
    }
    return internal_weights, diagnostics


def compute_commodity_engine_weights(prices, i, commodity_buckets):
    available = [b for b in commodity_buckets if has_lookback(prices, b, i)]
    raw_scores = {b: compute_score(prices, b, i) for b in available}
    adjusted_scores = {}
    for b, s in raw_scores.items():
        if pd.isna(s) or s <= 0:
            adjusted_scores[b] = 0.0
        else:
            adjusted_scores[b] = float(s) * (BROAD_COMMODITY_SCORE_BOOST if b == "broad_commodities" else 1.0)

    power_scores = {b: s ** COMMODITY_ENGINE_SCORE_POWER for b, s in adjusted_scores.items() if s > 0}
    if not power_scores:
        diagnostics = {
            "available_buckets": available,
            "bucket_scores": {k: None if pd.isna(v) else float(v) for k, v in raw_scores.items()},
            "adjusted_scores": adjusted_scores,
            "selected_buckets": [],
            "internal_weights": {},
            "score_power": COMMODITY_ENGINE_SCORE_POWER,
            "broad_score_boost": BROAD_COMMODITY_SCORE_BOOST,
            "max_total_commodities": MAX_TOTAL_COMMODITIES,
            "max_broad_commodities": MAX_BROAD_COMMODITIES,
            "max_single_commodity_sub": MAX_SINGLE_COMMODITY_SUB,
        }
        return {}, diagnostics

    weights = normalize_weights(power_scores)
    caps = {b: MAX_SINGLE_COMMODITY_SUB / MAX_TOTAL_COMMODITIES for b in weights}
    if "broad_commodities" in weights:
        caps["broad_commodities"] = MAX_BROAD_COMMODITIES / MAX_TOTAL_COMMODITIES
    weights = cap_and_redistribute(weights, caps)
    selected = sorted(weights, key=lambda b: weights[b], reverse=True)
    diagnostics = {
        "available_buckets": available,
        "bucket_scores": {k: None if pd.isna(v) else float(v) for k, v in raw_scores.items()},
        "adjusted_scores": adjusted_scores,
        "power_scores": {k: float(v) for k, v in power_scores.items()},
        "selected_buckets": selected,
        "internal_weights": {k: float(v) for k, v in weights.items()},
        "score_power": COMMODITY_ENGINE_SCORE_POWER,
        "broad_score_boost": BROAD_COMMODITY_SCORE_BOOST,
        "max_total_commodities": MAX_TOTAL_COMMODITIES,
        "max_broad_commodities": MAX_BROAD_COMMODITIES,
        "max_single_commodity_sub": MAX_SINGLE_COMMODITY_SUB,
    }
    return weights, diagnostics


def score_from_weights(prices, i, weights):
    vals = []
    for asset, weight in weights.items():
        s = compute_score(prices, asset, i)
        if pd.notna(s):
            vals.append(float(weight) * float(s))
    return float(sum(vals)) if vals else np.nan


def weighted_score(asset_weights, asset_scores):
    total = 0.0
    used = 0.0
    for asset, weight in asset_weights.items():
        score = 0.0 if asset == RISK_FREE_ASSET else asset_scores.get(asset, np.nan)
        if pd.notna(score):
            total += float(weight) * float(score)
            used += float(weight)
    return total / used if used > 0 else np.nan


def apply_equity_engine_minimum(w_step, asset_scores, benchmark_score, ranking):
    if EQUITY_ENGINE_ASSET not in w_step or EQUITY_ENGINE_ASSET not in ranking:
        return w_step, None
    eq_score = asset_scores.get(EQUITY_ENGINE_ASSET, np.nan)
    if pd.isna(eq_score) or pd.isna(benchmark_score) or eq_score <= benchmark_score:
        return w_step, None
    rank_pos = ranking.index(EQUITY_ENGINE_ASSET)
    target = None
    reason = None
    if rank_pos == 0 and eq_score > benchmark_score + EQUITY_ENGINE_STRONG_EDGE:
        target = EQUITY_ENGINE_MIN_STRONG_TOP1
        reason = "equity_engine_top1_strong_vs_benchmark"
    elif rank_pos == 0:
        target = EQUITY_ENGINE_MIN_TOP1
        reason = "equity_engine_top1_vs_benchmark"
    elif rank_pos == 1:
        target = EQUITY_ENGINE_MIN_TOP2
        reason = "equity_engine_top2_vs_benchmark"
    if target is None or w_step.get(EQUITY_ENGINE_ASSET, 0.0) >= target:
        return w_step, None
    current = w_step.get(EQUITY_ENGINE_ASSET, 0.0)
    need = target - current
    donors = {k: v for k, v in w_step.items() if k != EQUITY_ENGINE_ASSET and v > 0}
    donor_sum = sum(donors.values())
    if donor_sum <= 0:
        return w_step, None
    for k, v in donors.items():
        w_step[k] = max(0.0, w_step[k] - need * (v / donor_sum))
    w_step[EQUITY_ENGINE_ASSET] = target
    return normalize_weights(w_step), reason


def build_benchmark_asset_weights():
    return {EQUITY_ENGINE_ASSET: BENCHMARK_WEIGHTS["equities"], "bonds": BENCHMARK_WEIGHTS["bonds"]}


def build_v6_weights(prices, i, gics_buckets, commodity_buckets, include_bitcoin=True, equity_factor_max=DEFAULT_EQUITY_FACTOR_MAX):
    equity_engine_weights, equity_diag = compute_equity_engine_weights(prices, i, gics_buckets, factor_max=equity_factor_max)
    commodity_engine_weights, commodity_diag = compute_commodity_engine_weights(prices, i, commodity_buckets)

    configured_assets = [a for a in NEUTRAL_WEIGHTS if include_bitcoin or a != "bitcoin"]
    available_assets = []
    for asset in configured_assets:
        if asset == EQUITY_ENGINE_ASSET:
            if equity_engine_weights and all(has_lookback(prices, a, i) for a in equity_engine_weights):
                available_assets.append(asset)
        elif asset == COMMODITY_ENGINE_ASSET:
            if commodity_engine_weights and all(has_lookback(prices, a, i) for a in commodity_engine_weights):
                available_assets.append(asset)
        elif has_lookback(prices, asset, i):
            available_assets.append(asset)
    if not available_assets:
        raise ValueError("Keine verfügbaren Assets mit ausreichender Historie")

    rankable_assets = [a for a in available_assets if a != RISK_FREE_ASSET]
    asset_scores = {}
    for asset in rankable_assets:
        if asset == EQUITY_ENGINE_ASSET:
            asset_scores[asset] = score_from_weights(prices, i, equity_engine_weights)
        elif asset == COMMODITY_ENGINE_ASSET:
            asset_scores[asset] = score_from_weights(prices, i, commodity_engine_weights)
        else:
            asset_scores[asset] = compute_score(prices, asset, i)

    benchmark_score = compute_benchmark_score(prices, i)
    raw_ranking = sorted(asset_scores, key=lambda a: asset_scores[a] if pd.notna(asset_scores[a]) else -999, reverse=True)
    eligible_assets = [a for a in raw_ranking if pd.notna(asset_scores.get(a)) and pd.notna(benchmark_score) and asset_scores[a] > benchmark_score]
    ranking = eligible_assets if BENCHMARK_RELATIVE_FILTER else raw_ranking
    top3 = ranking[:3]

    if not top3:
        w_step = build_benchmark_asset_weights() if BENCHMARK_FALLBACK_MODE == "benchmark" else {RISK_FREE_ASSET: 1.0}
        fallback_used = True
        fallback_reason = "no_asset_outperforms_70_30_score"
    else:
        base = {a: NEUTRAL_WEIGHTS[a] for a in available_assets}
        base_sum = sum(base.values())
        w_step = {a: v / base_sum for a, v in base.items()}
        boosts = {0: 0.10, 1: 0.05, 2: 0.02}
        boosted_assets = set()
        for rank, asset in enumerate(top3):
            boost = boosts[rank]
            w_step[asset] = w_step.get(asset, 0.0) + boost
            boosted_assets.add(asset)
        total_extra = sum(boosts[r] for r in range(len(top3)))
        donors = {k: v for k, v in w_step.items() if k not in boosted_assets and v > 0}
        donor_sum = sum(donors.values())
        if donor_sum > 0 and total_extra > 0:
            for k, v in donors.items():
                w_step[k] = max(0.0, w_step[k] - total_extra * (v / donor_sum))
        w_step = normalize_weights(w_step)

        caps = {COMMODITY_ENGINE_ASSET: MAX_TOTAL_COMMODITIES}
        w_step = cap_and_redistribute(w_step, caps)
        w_step, min_reason = apply_equity_engine_minimum(w_step, asset_scores, benchmark_score, raw_ranking)
        fallback_used = False
        fallback_reason = min_reason
        portfolio_score = weighted_score(w_step, asset_scores)
        if pd.notna(portfolio_score) and pd.notna(benchmark_score) and portfolio_score <= benchmark_score:
            w_step = build_benchmark_asset_weights() if BENCHMARK_FALLBACK_MODE == "benchmark" else {RISK_FREE_ASSET: 1.0}
            fallback_used = True
            fallback_reason = "portfolio_score_not_above_70_30_score"

    current_weights = pd.Series(0.0, index=prices.columns)
    for asset, weight in w_step.items():
        if asset == EQUITY_ENGINE_ASSET:
            for inner_asset, inner_weight in equity_engine_weights.items():
                current_weights[inner_asset] = current_weights.get(inner_asset, 0.0) + weight * inner_weight
        elif asset == COMMODITY_ENGINE_ASSET:
            for inner_asset, inner_weight in commodity_engine_weights.items():
                current_weights[inner_asset] = current_weights.get(inner_asset, 0.0) + weight * inner_weight
        else:
            current_weights[asset] = current_weights.get(asset, 0.0) + weight

    if current_weights.sum() <= 0:
        raise ValueError("Gewichtssumme ist null")
    current_weights = current_weights / current_weights.sum()
    diagnostics = {
        "available_assets": available_assets,
        "asset_scores": {k: None if pd.isna(v) else float(v) for k, v in asset_scores.items()},
        "benchmark_score": None if pd.isna(benchmark_score) else float(benchmark_score),
        "raw_asset_ranking": raw_ranking,
        "eligible_assets_vs_benchmark": eligible_assets,
        "asset_ranking": ranking,
        "selected_assets": top3,
        "equity_engine": equity_diag,
        "commodity_engine": commodity_diag,
        "asset_level_weights": {k: float(v) for k, v in w_step.items()},
        "asset_level_portfolio_score": None if pd.isna(weighted_score(w_step, asset_scores)) else float(weighted_score(w_step, asset_scores)),
        "benchmark_relative_filter": BENCHMARK_RELATIVE_FILTER,
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
        "weighting_method": "v6_adds_score_weighted_commodity_engine_with_caps",
    }
    return current_weights, diagnostics


def run(prices, rets, settings, gics_buckets, commodity_buckets):
    start = resolve_start_index(prices, settings.get("start_date"))
    nav = [1.0]
    dates = [prices.index[start]]
    prev_w = pd.Series(0.0, index=prices.columns)
    history = []
    for i in range(start, len(prices) - 1):
        w, diagnostics = build_v6_weights(prices, i, gics_buckets, commodity_buckets, include_bitcoin=settings["include_bitcoin"], equity_factor_max=settings["equity_factor_max"])
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
            "v6": diagnostics,
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
    return {"strategy": label, "cagr": f"{cagr * 100:.1f} %", "vola": f"{vola * 100:.1f} %", "max_dd": f"{dd * 100:.1f} %", "sharpe": round(float(sharpe), 2)}


@app.route("/api/backtest_v6", methods=["POST"])
def backtest_v6():
    try:
        data = request.json or {}
        settings = {
            "include_bitcoin": bool(data.get("include_bitcoin", True)),
            "cost": float(data.get("transaction_cost_rate", DEFAULT_COST_RATE)),
            "start_date": data.get("start_date"),
            "equity_factor_max": clamp_factor_max(data.get("equity_factor_max", DEFAULT_EQUITY_FACTOR_MAX)),
            "sector_proxy_set": resolve_sector_proxy_set(data.get("sector_proxy_set", DEFAULT_SECTOR_PROXY_SET)),
        }
        prices, rets, gics_buckets, commodity_buckets = load_data(settings["include_bitcoin"], settings["sector_proxy_set"])
        nav, history = run(prices, rets, settings, gics_buckets, commodity_buckets)
        if "equities" not in rets.columns or "bonds" not in rets.columns:
            raise ValueError("Benchmark-Daten fehlen")
        bench_ret = rets[["equities", "bonds"]].dropna().mul(pd.Series(BENCHMARK_WEIGHTS), axis=1).sum(axis=1)
        bench = (1.0 + bench_ret[bench_ret.index >= nav.index[0]]).cumprod()
        idx = nav.index.intersection(bench.index)
        if len(idx) == 0:
            raise ValueError("Keine gemeinsame Historie zwischen Portfolio und Benchmark")

        assets = ["equities", "bonds", "managed_futures", "gold", "cash"]
        if settings["include_bitcoin"]:
            assets.append("bitcoin")
        availability = compute_availability(prices, assets, list(gics_buckets.keys()) + list(commodity_buckets.keys()))
        proxies = {asset: DEFAULT_TICKERS[asset] for asset in assets if asset in DEFAULT_TICKERS}
        proxies["managed_futures"] = "data/SG_TRD_IDX.TXT" if SG_TREND_FILE.exists() else DEFAULT_TICKERS["managed_futures"]
        proxies.update({bucket: ticker for bucket, ticker in gics_buckets.items()})
        proxies.update({bucket: ticker for bucket, ticker in commodity_buckets.items()})

        return jsonify({
            "summary": [get_stats(nav.loc[idx], "Portfolio V6"), get_stats(bench.loc[idx], "Benchmark 70/30")],
            "chart": {"dates": [d.strftime("%Y-%m-%d") for d in idx], "portfolio": nav.loc[idx].tolist(), "benchmark": bench.loc[idx].tolist()},
            "latest_weights": history[-1] if history else None,
            "history": history,
            "meta": {
                "version": "v6_commodity_engine",
                "include_bitcoin": settings["include_bitcoin"],
                "transaction_cost_rate": settings["cost"],
                "equity_factor_max": settings["equity_factor_max"],
                "equity_factor_max_allowed": MAX_EQUITY_FACTOR_MAX,
                "equity_engine_top_n": EQUITY_ENGINE_TOP_N,
                "equity_engine_weighting": "base equities plus top sectors/factors; 70% equal + 30% edge^1.5",
                "commodity_engine_weighting": "positive commodity scores only; broad commodity score boost; score^1.5; caps and redistribution",
                "commodity_engine_score_power": COMMODITY_ENGINE_SCORE_POWER,
                "broad_commodity_score_boost": BROAD_COMMODITY_SCORE_BOOST,
                "max_total_commodities": MAX_TOTAL_COMMODITIES,
                "max_broad_commodities": MAX_BROAD_COMMODITIES,
                "max_single_commodity_sub": MAX_SINGLE_COMMODITY_SUB,
                "gold_treatment": "gold remains separate from commodity_engine",
                "sector_proxy_set": settings["sector_proxy_set"],
                "allowed_sector_proxy_sets": sorted(ALLOWED_SECTOR_PROXY_SETS),
                "start_date": settings["start_date"],
                "actual_start_date": idx[0].strftime("%Y-%m-%d"),
                "data_mode": "rolling_available_universe",
                "missing_asset_method": "method_a_available_assets_sum_to_100",
                "minimum_core_assets": MIN_CORE_ASSETS,
                "lookback_months": LOOKBACK_MONTHS,
                "ranking_method": "v6: equity_engine and commodity_engine are built first; asset-level ranking keeps only assets whose score beats 70/30 score; final asset portfolio must also beat 70/30 score",
                "weighting_method": "v6_adds_score_weighted_commodity_engine_with_caps",
                "benchmark_relative_filter": BENCHMARK_RELATIVE_FILTER,
                "benchmark_fallback_mode": BENCHMARK_FALLBACK_MODE,
                "managed_futures_source": "local_sg_trend_file" if SG_TREND_FILE.exists() else "yfinance_dbmf",
                "available_from": availability,
                "proxies": proxies,
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "portfolio-webapp", "version": "v6", "endpoint": "/api/backtest_v6"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
