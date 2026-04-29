import math
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

START_DATE = "2000-01-01"
LOOKBACK_MONTHS = 12
DEFAULT_COST_RATE = 0.001
DEFAULT_SECTOR_PROXY_SET = "us_long_history"
ALLOWED_SECTOR_PROXY_SETS = {"ucits", "us_long_history"}

DEFAULT_TICKERS = {
    "equities": "ACWI",
}

GICS_BUCKETS_UCITS = {
    "energy": "WNRG.DE",
    "materials": "XMWS.DE",
    "defense": "DFND.L",
    "industrials_ex": "WIND.DE",
    "cons_disc": "SC0G.DE",
    "cons_staples": "WCSS.DE",
    "pharma_bio": "BIOT.DE",
    "hc_equip": "IHI",
    "banks": "EXV1.DE",
    "fin_serv": "WFIN.DE",
    "software": "IGPT.DE",
    "semis": "VVSM.DE",
    "comm_serv": "XWTS.DE",
    "utilities": "WUTI.DE",
    "real_estate": "DPRE.DE",
}

GICS_BUCKETS_US_LONG_HISTORY = {
    "energy": "XLE",
    "materials": "XLB",
    "industrials": "XLI",
    "cons_disc": "XLY",
    "cons_staples": "XLP",
    "health_care": "XLV",
    "financials": "XLF",
    "technology": "XLK",
    "utilities": "XLU",
    "real_estate": "IYR",
    "semis": "SMH",
    "biotech": "IBB",
    "hc_equip": "IHI",
    "banks": "KBE",
    "aerospace_defense": "ITA",
    "infrastructure": "PAVE",
    "homebuilders": "XHB",
    "transportation": "IYT",
    "gold_miners": "GDX",
    "oil_services": "OIH",
    "metals_mining": "XME",
    "software": "IGV",
}

SECTOR_PROXY_SETS = {
    "ucits": GICS_BUCKETS_UCITS,
    "us_long_history": GICS_BUCKETS_US_LONG_HISTORY,
}


def get_sector_buckets(sector_proxy_set):
    if sector_proxy_set not in SECTOR_PROXY_SETS:
        raise ValueError("sector_proxy_set muss ucits oder us_long_history sein")
    return SECTOR_PROXY_SETS[sector_proxy_set]


def resolve_sector_proxy_set(value):
    if value is None:
        return DEFAULT_SECTOR_PROXY_SET
    value = str(value).strip()
    if value not in ALLOWED_SECTOR_PROXY_SETS:
        raise ValueError("sector_proxy_set muss ucits oder us_long_history sein")
    return value


def load_data(sector_proxy_set=DEFAULT_SECTOR_PROXY_SET):
    gics_buckets = get_sector_buckets(sector_proxy_set)
    tickers = sorted(set([DEFAULT_TICKERS["equities"]] + list(gics_buckets.values())))

    data = yf.download(tickers, start=START_DATE, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(tickers[0])

    series = {}
    if DEFAULT_TICKERS["equities"] in data.columns:
        series["equities"] = pd.to_numeric(data[DEFAULT_TICKERS["equities"]], errors="coerce").resample("ME").last()

    for bucket, ticker in gics_buckets.items():
        if ticker in data.columns:
            series[bucket] = pd.to_numeric(data[ticker], errors="coerce").resample("ME").last()

    if "equities" not in series:
        raise ValueError("ACWI-Daten konnten nicht geladen werden")

    prices = pd.concat(series, axis=1).sort_index().ffill()
    prices = prices[prices.index >= pd.to_datetime(START_DATE)]
    rets = prices.pct_change()
    return prices, rets, gics_buckets


def has_lookback(prices, asset, i, lookback=LOOKBACK_MONTHS):
    if asset not in prices.columns or i < lookback:
        return False
    window = prices[asset].iloc[i - lookback:i + 1]
    return not window.isna().any()


def compute_score(prices, asset, i):
    if not has_lookback(prices, asset, i):
        return np.nan
    p = prices[asset]
    p_now = p.iloc[i]
    p3 = p.iloc[i - 3]
    p6 = p.iloc[i - 6]
    p12 = p.iloc[i - 12]
    if any(pd.isna(v) or v == 0 for v in [p_now, p3, p6, p12]):
        return np.nan
    return float(0.5 * (p_now / p3 - 1) + 0.3 * (p_now / p6 - 1) + 0.2 * (p_now / p12 - 1))


def normalize_weights(weights):
    weights = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(weights.values())
    if total <= 0:
        return weights
    return {k: v / total for k, v in weights.items()}


def cap_and_redistribute(weights, caps, max_iter=20):
    weights = normalize_weights(weights)
    locked = set()
    for _ in range(max_iter):
        changed = False
        excess = 0.0
        for k, w in list(weights.items()):
            cap = caps.get(k)
            if cap is not None and w > cap:
                excess += w - cap
                weights[k] = cap
                locked.add(k)
                changed = True
        if not changed or excess <= 1e-12:
            break
        free_total = sum(w for k, w in weights.items() if k not in locked)
        if free_total <= 0:
            break
        for k, w in list(weights.items()):
            if k not in locked:
                weights[k] = w + excess * (w / free_total)
    return normalize_weights(weights)


def compute_equity_engine_weights(
    prices,
    i,
    gics_buckets,
    factor_max=0.60,
    top_n=3,
    score_power=1.5,
    equal_weight_blend=0.60,
    edge_weight_blend=0.40,
    min_edge=0.00,
    min_abs_score=0.00,
    max_single_sector=0.25,
):
    equity_score = compute_score(prices, "equities", i)
    if pd.isna(equity_score):
        return {"equities": 1.0}, {"reason": "no_equity_score", "selected_buckets": []}

    available = [b for b in gics_buckets if has_lookback(prices, b, i)]
    bucket_scores = {b: compute_score(prices, b, i) for b in available}

    candidates = [
        b for b in available
        if pd.notna(bucket_scores.get(b))
        and bucket_scores[b] > equity_score + min_edge
        and bucket_scores[b] > min_abs_score
    ]
    top_buckets = sorted(candidates, key=lambda b: bucket_scores[b], reverse=True)[:top_n]

    if not top_buckets:
        return {"equities": 1.0}, {
            "reason": "no_sector_above_acwi",
            "equity_score": float(equity_score),
            "selected_buckets": [],
            "bucket_scores": {k: None if pd.isna(v) else float(v) for k, v in bucket_scores.items()},
        }

    edges = {b: max(0.0, float(bucket_scores[b] - equity_score)) for b in top_buckets}
    avg_edge = float(np.mean(list(edges.values())))

    if avg_edge < 0.02:
        factor_weight = factor_max * 0.50
    elif avg_edge < 0.05:
        factor_weight = factor_max * 0.75
    else:
        factor_weight = factor_max

    if equity_score < 0:
        factor_weight *= 0.50

    base_weight = 1.0 - factor_weight
    equal_weights = {b: factor_weight / len(top_buckets) for b in top_buckets}
    edge_power_scores = {b: edges[b] ** score_power for b in top_buckets if edges[b] > 0}
    total_power = sum(edge_power_scores.values())

    if total_power > 0:
        edge_weights = {b: factor_weight * edge_power_scores.get(b, 0.0) / total_power for b in top_buckets}
    else:
        edge_weights = dict(equal_weights)

    sector_weights = {
        b: equal_weight_blend * equal_weights.get(b, 0.0) + edge_weight_blend * edge_weights.get(b, 0.0)
        for b in top_buckets
    }

    weights = {"equities": base_weight, **sector_weights}
    weights = cap_and_redistribute(weights, {b: max_single_sector for b in top_buckets})

    diag = {
        "reason": "sector_overlay_active",
        "equity_score": float(equity_score),
        "selected_buckets": top_buckets,
        "edges": {k: float(v) for k, v in edges.items()},
        "avg_edge": avg_edge,
        "factor_weight": float(factor_weight),
        "base_weight": float(base_weight),
        "bucket_scores": {k: None if pd.isna(v) else float(v) for k, v in bucket_scores.items()},
        "internal_weights": {k: float(v) for k, v in weights.items()},
    }
    return weights, diag


def max_drawdown(equity_curve):
    equity_curve = pd.Series(equity_curve).dropna()
    if equity_curve.empty:
        return np.nan
    return float((equity_curve / equity_curve.cummax() - 1.0).min())


def calc_stats(monthly_returns):
    monthly_returns = pd.Series(monthly_returns).dropna()
    if monthly_returns.empty:
        return {"cagr": None, "vola": None, "max_drawdown": None, "sharpe": None, "total_return": None}
    curve = (1.0 + monthly_returns).cumprod()
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    cagr = float(curve.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    vola = float(monthly_returns.std() * math.sqrt(12))
    sharpe = float(cagr / vola) if vola > 0 else np.nan
    return {
        "cagr": None if pd.isna(cagr) else cagr,
        "vola": None if pd.isna(vola) else vola,
        "max_drawdown": max_drawdown(curve),
        "sharpe": None if pd.isna(sharpe) else sharpe,
        "total_return": float(curve.iloc[-1] - 1.0),
    }


def run_backtest(
    start_date=None,
    sector_proxy_set=DEFAULT_SECTOR_PROXY_SET,
    cost_rate=DEFAULT_COST_RATE,
    factor_max=0.60,
    top_n=3,
    score_power=1.5,
    equal_weight_blend=0.60,
    edge_weight_blend=0.40,
    min_edge=0.00,
    min_abs_score=0.00,
    max_single_sector=0.25,
):
    prices, rets, gics_buckets = load_data(sector_proxy_set)

    if start_date:
        target = pd.to_datetime(start_date, errors="coerce")
        if pd.isna(target):
            raise ValueError("Ungültiges Startdatum. Format bitte YYYY-MM-DD")
        candidate_positions = np.where(prices.index >= target)[0]
    else:
        candidate_positions = range(LOOKBACK_MONTHS, len(prices))

    start_i = None
    for i in candidate_positions:
        if has_lookback(prices, "equities", int(i)):
            start_i = int(i)
            break
    if start_i is None:
        raise ValueError("Nicht genügend Historie für ACWI")

    engine_returns = []
    acwi_returns = []
    dates = []
    weight_history = []
    previous_weights = None

    for i in range(start_i, len(prices) - 1):
        weights, diag = compute_equity_engine_weights(
            prices, i, gics_buckets,
            factor_max=factor_max,
            top_n=top_n,
            score_power=score_power,
            equal_weight_blend=equal_weight_blend,
            edge_weight_blend=edge_weight_blend,
            min_edge=min_edge,
            min_abs_score=min_abs_score,
            max_single_sector=max_single_sector,
        )

        if previous_weights is None:
            turnover = sum(abs(v) for v in weights.values())
        else:
            all_assets = set(previous_weights.keys()) | set(weights.keys())
            turnover = sum(abs(weights.get(a, 0.0) - previous_weights.get(a, 0.0)) for a in all_assets)
        transaction_cost = turnover * cost_rate

        month_return = 0.0
        for asset, weight in weights.items():
            if asset in rets.columns:
                r = rets[asset].iloc[i + 1]
                if pd.notna(r):
                    month_return += float(weight) * float(r)
        month_return -= transaction_cost

        acwi_return = rets["equities"].iloc[i + 1]
        acwi_return = 0.0 if pd.isna(acwi_return) else float(acwi_return)

        date = prices.index[i + 1]
        dates.append(date)
        engine_returns.append(float(month_return))
        acwi_returns.append(acwi_return)

        weight_history.append({
            "date": date.strftime("%Y-%m-%d"),
            **{k: float(v) for k, v in weights.items()},
            "turnover": float(turnover),
            "transaction_cost": float(transaction_cost),
            "selected_buckets": diag.get("selected_buckets", []),
            "reason": diag.get("reason"),
        })
        previous_weights = weights

    engine_returns = pd.Series(engine_returns, index=pd.DatetimeIndex(dates), name="equity_engine")
    acwi_returns = pd.Series(acwi_returns, index=pd.DatetimeIndex(dates), name="acwi")
    engine_curve = (1.0 + engine_returns).cumprod()
    acwi_curve = (1.0 + acwi_returns).cumprod()
    excess_returns = engine_returns - acwi_returns

    selected_counts = {}
    active_overlay_months = 0
    for row in weight_history:
        selected = row.get("selected_buckets", [])
        if selected:
            active_overlay_months += 1
        for b in selected:
            selected_counts[b] = selected_counts.get(b, 0) + 1
    selected_counts = dict(sorted(selected_counts.items(), key=lambda x: x[1], reverse=True))

    tracking_error = float(excess_returns.std() * math.sqrt(12)) if len(excess_returns) > 1 else None
    information_ratio = float((excess_returns.mean() * 12) / tracking_error) if tracking_error and tracking_error > 0 else None

    return {
        "strategy": "optimized_equity_engine_vs_acwi",
        "start_date": engine_returns.index[0].strftime("%Y-%m-%d") if len(engine_returns) else None,
        "end_date": engine_returns.index[-1].strftime("%Y-%m-%d") if len(engine_returns) else None,
        "sector_proxy_set": sector_proxy_set,
        "cost_rate": float(cost_rate),
        "params": {
            "factor_max": float(factor_max),
            "top_n": int(top_n),
            "score_power": float(score_power),
            "equal_weight_blend": float(equal_weight_blend),
            "edge_weight_blend": float(edge_weight_blend),
            "min_edge": float(min_edge),
            "min_abs_score": float(min_abs_score),
            "max_single_sector": float(max_single_sector),
        },
        "performance": {
            "equity_engine": calc_stats(engine_returns),
            "acwi": calc_stats(acwi_returns),
            "excess": {
                "hit_rate": float((excess_returns > 0).mean()) if len(excess_returns) else None,
                "avg_monthly_excess": float(excess_returns.mean()) if len(excess_returns) else None,
                "tracking_error": tracking_error,
                "information_ratio": information_ratio,
            },
        },
        "series": {
            "dates": [d.strftime("%Y-%m-%d") for d in engine_curve.index],
            "equity_engine": [float(x) for x in engine_curve.values],
            "acwi": [float(x) for x in acwi_curve.values],
        },
        "latest_weights": weight_history[-1] if weight_history else {},
        "selection_counts": selected_counts,
        "overlay_usage": {
            "active_overlay_months": int(active_overlay_months),
            "total_months": int(len(weight_history)),
            "active_overlay_share": float(active_overlay_months / len(weight_history)) if weight_history else None,
        },
        "weight_history": weight_history,
    }


@app.route("/api/equity-engine", methods=["GET", "POST"])
def api_equity_engine():
    try:
        payload = request.get_json(silent=True) or {}
        def get_param(name, default):
            return payload.get(name, request.args.get(name, default))

        result = run_backtest(
            start_date=get_param("start_date", None),
            sector_proxy_set=resolve_sector_proxy_set(get_param("sector_proxy_set", DEFAULT_SECTOR_PROXY_SET)),
            cost_rate=float(get_param("cost_rate", DEFAULT_COST_RATE)),
            factor_max=float(get_param("factor_max", 0.60)),
            top_n=int(get_param("top_n", 3)),
            score_power=float(get_param("score_power", 1.5)),
            equal_weight_blend=float(get_param("equal_weight_blend", 0.60)),
            edge_weight_blend=float(get_param("edge_weight_blend", 0.40)),
            min_edge=float(get_param("min_edge", 0.00)),
            min_abs_score=float(get_param("min_abs_score", 0.00)),
            max_single_sector=float(get_param("max_single_sector", 0.25)),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/", methods=["GET"])
def index():
    return Response(FRONTEND_HTML, mimetype="text/html")


FRONTEND_HTML = r"""
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Equity Engine Test</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body { font-family: Arial, sans-serif; color:#222; margin:0; background:#f6f6f6; }
    .wrap { max-width: 1150px; margin: 32px auto; padding: 0 16px; }
    .box { background:#fff; border:1px solid #ddd; border-radius:10px; padding:18px; margin-bottom:18px; }
    .row { display:flex; gap:14px; flex-wrap:wrap; align-items:end; }
    .field { display:flex; flex-direction:column; min-width:160px; }
    label { font-size:13px; margin-bottom:5px; color:#555; }
    input, select { padding:9px; border:1px solid #bbb; border-radius:6px; }
    button { padding:10px 16px; border:0; border-radius:6px; background:#222; color:white; cursor:pointer; }
    table { border-collapse:collapse; width:100%; margin-top:10px; }
    th, td { border:1px solid #ddd; padding:9px; text-align:left; font-size:14px; }
    th { background:#eee; }
    .muted { color:#777; font-size:13px; }
    .grid { display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; }
    .metric { background:#fafafa; border:1px solid #ddd; border-radius:8px; padding:12px; }
    .metric b { display:block; font-size:13px; color:#666; margin-bottom:6px; }
    .metric span { font-size:20px; }
    @media(max-width:800px){ .grid { grid-template-columns:1fr 1fr; } }
  </style>
</head>
<body>
<div class="wrap">
  <h1>Equity Engine Test vs. ACWI</h1>
  <p class="muted">Separater Test: reine Equity Engine gegen ACWI. Keine Bonds, kein Gold, kein Bitcoin, keine Commodity Engine.</p>

  <div class="box">
    <div class="row">
      <div class="field"><label>Startdatum</label><input id="start_date" value="2009-03-31" /></div>
      <div class="field"><label>Sektor-Set</label><select id="sector_proxy_set"><option value="us_long_history">US Long History</option><option value="ucits">UCITS</option></select></div>
      <div class="field"><label>Factor Max</label><input id="factor_max" type="number" step="0.05" value="0.60" /></div>
      <div class="field"><label>Top N</label><input id="top_n" type="number" step="1" value="3" /></div>
      <div class="field"><label>Max Einzelsektor</label><input id="max_single_sector" type="number" step="0.05" value="0.25" /></div>
      <div class="field"><label>Score Power</label><input id="score_power" type="number" step="0.1" value="1.5" /></div>
      <div class="field"><label>Kosten pro Turnover</label><input id="cost_rate" type="number" step="0.0005" value="0.001" /></div>
      <button onclick="runTest()">Test starten</button>
    </div>
  </div>

  <div class="box"><canvas id="chart" height="110"></canvas></div>

  <div class="box">
    <h2>Performance</h2>
    <table id="perfTable"></table>
  </div>

  <div class="box">
    <h2>Excess / Qualität</h2>
    <div class="grid" id="qualityGrid"></div>
  </div>

  <div class="box">
    <h2>Letzte Gewichtung</h2>
    <table id="weightsTable"></table>
  </div>

  <div class="box">
    <h2>Häufigste Sektor-Auswahl</h2>
    <table id="selectionTable"></table>
  </div>

  <div class="box">
    <h2>Monatliche Gewichtungshistorie</h2>
    <table id="historyTable"></table>
  </div>
</div>

<script>
let chart;
function pct(x){ return x === null || x === undefined ? "-" : (x*100).toFixed(1)+" %"; }
function num(x){ return x === null || x === undefined ? "-" : Number(x).toFixed(2); }

async function runTest(){
  const params = new URLSearchParams({
    start_date: document.getElementById('start_date').value,
    sector_proxy_set: document.getElementById('sector_proxy_set').value,
    factor_max: document.getElementById('factor_max').value,
    top_n: document.getElementById('top_n').value,
    max_single_sector: document.getElementById('max_single_sector').value,
    score_power: document.getElementById('score_power').value,
    cost_rate: document.getElementById('cost_rate').value
  });
  const res = await fetch('/api/equity-engine?' + params.toString());
  const data = await res.json();
  if(data.error){ alert(data.error); return; }
  render(data);
}

function render(data){
  const labels = data.series.dates;
  const ctx = document.getElementById('chart');
  if(chart) chart.destroy();
  chart = new Chart(ctx, {
    type:'line',
    data:{ labels, datasets:[
      { label:'Equity Engine', data:data.series.equity_engine, borderWidth:2, pointRadius:0 },
      { label:'ACWI', data:data.series.acwi, borderWidth:2, pointRadius:0 }
    ]},
    options:{ responsive:true, interaction:{mode:'index', intersect:false}, scales:{ y:{ beginAtZero:false } } }
  });

  const p = data.performance;
  document.getElementById('perfTable').innerHTML = `
    <tr><th>Strategie</th><th>CAGR</th><th>Vola</th><th>Max DD</th><th>Sharpe</th><th>Total Return</th></tr>
    <tr><td>Equity Engine</td><td>${pct(p.equity_engine.cagr)}</td><td>${pct(p.equity_engine.vola)}</td><td>${pct(p.equity_engine.max_drawdown)}</td><td>${num(p.equity_engine.sharpe)}</td><td>${pct(p.equity_engine.total_return)}</td></tr>
    <tr><td>ACWI</td><td>${pct(p.acwi.cagr)}</td><td>${pct(p.acwi.vola)}</td><td>${pct(p.acwi.max_drawdown)}</td><td>${num(p.acwi.sharpe)}</td><td>${pct(p.acwi.total_return)}</td></tr>`;

  document.getElementById('qualityGrid').innerHTML = `
    <div class="metric"><b>Hit Rate vs. ACWI</b><span>${pct(p.excess.hit_rate)}</span></div>
    <div class="metric"><b>Ø Monats-Excess</b><span>${pct(p.excess.avg_monthly_excess)}</span></div>
    <div class="metric"><b>Tracking Error</b><span>${pct(p.excess.tracking_error)}</span></div>
    <div class="metric"><b>Information Ratio</b><span>${num(p.excess.information_ratio)}</span></div>
    <div class="metric"><b>Overlay Monate</b><span>${data.overlay_usage.active_overlay_months}/${data.overlay_usage.total_months}</span></div>
    <div class="metric"><b>Overlay Anteil</b><span>${pct(data.overlay_usage.active_overlay_share)}</span></div>`;

  const latest = data.latest_weights || {};
  const weightRows = Object.keys(latest).filter(k => typeof latest[k] === 'number' && k !== 'turnover' && k !== 'transaction_cost').map(k => `<tr><td>${k}</td><td>${pct(latest[k])}</td></tr>`).join('');
  document.getElementById('weightsTable').innerHTML = `<tr><th>Asset</th><th>Gewicht</th></tr>${weightRows}<tr><td>Turnover</td><td>${pct(latest.turnover)}</td></tr><tr><td>Kosten</td><td>${pct(latest.transaction_cost)}</td></tr>`;

  const selRows = Object.entries(data.selection_counts).map(([k,v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('');
  document.getElementById('selectionTable').innerHTML = `<tr><th>Sektor</th><th>Monate ausgewählt</th></tr>${selRows}`;

  const hist = data.weight_history.slice(-36).reverse();
  const histRows = hist.map(r => `<tr><td>${r.date}</td><td>${pct(r.equities || 0)}</td><td>${(r.selected_buckets || []).join(', ')}</td><td>${pct(r.turnover)}</td><td>${r.reason || ''}</td></tr>`).join('');
  document.getElementById('historyTable').innerHTML = `<tr><th>Datum</th><th>ACWI</th><th>Sektoren</th><th>Turnover</th><th>Grund</th></tr>${histRows}`;
}

runTest();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    port = int(__import__("os").environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
