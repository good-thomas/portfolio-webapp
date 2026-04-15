import os
import math
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, request, jsonify
from flask_cors import CORS
from dataclasses import dataclass

app = Flask(__name__)
CORS(app)  # Erlaubt deinem Frontend den Zugriff auf die API

# ------------------------------------------------------------
# KONFIGURATION & DATENLADEN
# ------------------------------------------------------------
START_DATE = "2000-01-01"
DEFAULT_TICKERS = {
    "equities": "SPY",
    "bonds": "VBMFX",
    "gold": "GC=F",
    "commodities": "^SPGSCI",
    "cash": "^IRX",
    "crypto": "BTC-USD"
}

BENCHMARK_WEIGHTS = {"equities": 0.70, "bonds": 0.30}

def load_data():
    returns = {}
    for asset, ticker in DEFAULT_TICKERS.items():
        data = yf.download(ticker, start=START_DATE, auto_adjust=True, progress=False)
        if data.empty: continue
        
        # Close Preise extrahieren
        s = data.iloc[:, 0] if isinstance(data.columns, pd.MultiIndex) else data['Close']
        
        if asset == "cash":
            # ^IRX ist annualisierte Rendite in %
            returns[asset] = (s.resample("QE").last() / 100.0) / 4.0
        else:
            returns[asset] = s.resample("QE").last().pct_change()
            
    df = pd.concat(returns, axis=1).fillna(0.0)
    return df.iloc[1:]

# ------------------------------------------------------------
# BACKTEST LOGIK
# ------------------------------------------------------------

def infer_pairwise_scores(next_returns, accuracy, rng, valid_assets):
    wins = pd.Series(0.0, index=valid_assets)
    assets = list(valid_assets)
    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            a, b = assets[i], assets[j]
            ra, rb = next_returns[a], next_returns[b]
            correct_winner = a if ra > rb else b
            predicted_winner = correct_winner if rng.random() < accuracy else (b if correct_winner == a else a)
            wins[predicted_winner] += 1.0
    return wins

def run_simulation(asset_returns, settings):
    rng = np.random.default_rng(42)
    dates = asset_returns.index
    assets = asset_returns.columns.tolist()
    
    # Ergebnisse sammeln
    all_sim_paths = []
    last_weights_df = None

    for sim_i in range(settings['n_simulations']):
        nav = [1.0]
        nav_dates = [dates[0]]
        current_weights = pd.Series(1.0 / len(assets), index=assets)
        weight_records = []
        
        window = 8
        for t in range(window, len(dates) - 1):
            trailing = asset_returns.iloc[t-window:t]
            next_q = asset_returns.iloc[t+1]
            
            trailing_vol = trailing.std(ddof=1).fillna(0)
            valid_now = trailing_vol[trailing_vol > 0.0001].index.tolist()
            
            # 1. Signal & Ranking
            scores = infer_pairwise_scores(next_q, settings['signal_accuracy'], rng, valid_now)
            
            # 2. Gewichte bauen (Top N)
            top_n = scores.nlargest(settings['top_n_assets']).index
            target_w = pd.Series(0.0, index=assets)
            if len(top_n) > 0:
                inv_vol = 1.0 / trailing_vol[top_n].replace(0, 0.01)
                weights = (scores[top_n] + 1) * inv_vol
                target_w[top_n] = weights / weights.sum()
            
            # 3. Risk Overlay
            bench_vol = trailing[["equities", "bonds"]].mul(pd.Series(BENCHMARK_WEIGHTS), axis=1).sum(axis=1).std(ddof=1)
            port_vol = np.sqrt(target_w.values @ trailing.cov().fillna(0).values @ target_w.values)
            leverage = min(1.0, (bench_vol * settings['risk_target_ratio']) / port_vol) if port_vol > 0 else 1.0
            
            final_w = target_w * leverage
            if "cash" in final_w: final_w["cash"] += 1.0 - final_w.sum()
            final_w = final_w.clip(lower=0)
            
            # Performance berechnen
            ret = (final_w * next_q).sum()
            nav.append(nav[-1] * (1 + ret))
            nav_dates.append(dates[t+1])
            
            if sim_i == 0: # Wir speichern Gewichte nur für die erste Sim (Repräsentativ)
                weight_records.append(final_w.to_dict())
                weight_records[-1]['date'] = dates[t+1].strftime('%Y-%m-%d')
        
        all_sim_paths.append(pd.Series(nav, index=nav_dates))
        if sim_i == 0:
            last_weights_df = weight_records

    # Median Pfad berechnen
    paths_df = pd.DataFrame(all_sim_paths).T.ffill()
    median_path = paths_df.median(axis=1)
    
    return median_path, last_weights_df

# ------------------------------------------------------------
# API ENDPUNKTE
# ------------------------------------------------------------

@app.route('/api/backtest', methods=['POST'])
def backtest():
    data = request.json
    settings = {
        'signal_accuracy': data.get('signal_accuracy', 0.8),
        'top_n_assets': data.get('top_n_assets', 3),
        'risk_target_ratio': data.get('risk_target_ratio', 0.9),
        'n_simulations': data.get('n_simulations', 100)
    }

    # Daten laden & Backtest ausführen
    df_returns = load_data()
    median_nav, weights_history = run_simulation(df_returns, settings)
    
    # Benchmark berechnen
    bench_ret = df_returns[["equities", "bonds"]].mul(pd.Series(BENCHMARK_WEIGHTS), axis=1).sum(axis=1)
    bench_nav = (1 + bench_ret[bench_ret.index >= median_nav.index[0]]).cumprod()
    bench_nav = (bench_nav / bench_nav.iloc[0]).tolist()

    # Kennzahlen (Beispiel für den Median-Pfad)
    total_ret = median_nav.pct_change().dropna()
    summary = [{
        "strategy": "Portfolio Median",
        "cagr": float((median_nav.iloc[-1]**(4/len(total_ret)))-1),
        "ann_vol": float(total_ret.std() * math.sqrt(4)),
        "max_drawdown": float((median_nav / median_nav.cummax() - 1).min()),
        "sharpe": float((total_ret.mean() / total_ret.std()) * math.sqrt(4)) if total_ret.std() != 0 else 0
    }]

    return jsonify({
        "summary": summary,
        "chart": {
            "dates": [d.strftime('%Y-%m-%d') for d in median_nav.index],
            "portfolio_median": median_nav.tolist(),
            "benchmark": bench_nav
        },
        "weights": weights_history
    })

if __name__ == '__main__':
    # Port für Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
