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
        
        # Robustes Auslesen der Close-Preise
        if isinstance(data.columns, pd.MultiIndex):
            s = data.iloc[:, 0] 
        else:
            s = data['Close']
            
        if asset == "cash":
            returns[asset] = (s.resample("QE").last() / 100.0) / 4.0
        else:
            returns[asset] = s.resample("QE").last().pct_change()
            
    df = pd.concat(returns, axis=1)
    df.columns = list(returns.keys())
    return df.dropna(how='all').fillna(0.0)

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
    all_sim_paths = []
    last_weights_df = None

    for sim_i in range(settings['n_simulations']):
        nav = [1.0]
        nav_dates = [dates[0]]
        weight_records = []
        window = 8
        
        for t in range(window, len(dates) - 1):
            trailing = asset_returns.iloc[t-window:t]
            next_q = asset_returns.iloc[t+1]
            trailing_vol = trailing.std(ddof=1).fillna(0)
            valid_now = trailing_vol[trailing_vol > 0.0001].index.tolist()
            
            scores = infer_pairwise_scores(next_q, settings['signal_accuracy'], rng, valid_now)
            top_n = scores.nlargest(settings['top_n_assets']).index
            
            target_w = pd.Series(0.0, index=assets)
            if len(top_n) > 0:
                inv_vol = 1.0 / trailing_vol[top_n].replace(0, 0.01)
                weights = (scores[top_n] + 1) * inv_vol
                target_w[top_n] = weights / weights.sum()
            
            # Risk Overlay gegen Benchmark (70/30)
            bench_vol = trailing[["equities", "bonds"]].mul(pd.Series(BENCHMARK_WEIGHTS), axis=1).sum(axis=1).std(ddof=1)
            port_vol = np.sqrt(target_w.values @ trailing.cov().fillna(0).values @ target_w.values)
            leverage = min(1.0, (bench_vol * settings['risk_target_ratio']) / port_vol) if port_vol > 0 else 1.0
            
            final_w = target_w * leverage
            if "cash" in final_w: final_w["cash"] += 1.0 - final_w.sum()
            final_w = final_w.clip(lower=0)
            
            ret = (final_w * next_q).sum()
            nav.append(nav[-1] * (1 + ret))
            nav_dates.append(dates[t+1])
            
            if sim_i == 0:
                weight_records.append({**final_w.to_dict(), 'date': dates[t+1].strftime('%Y-%m-%d')})
        
        all_sim_paths.append(pd.Series(nav, index=nav_dates))
        if sim_i == 0:
            last_weights_df = weight_records

    return pd.DataFrame(all_sim_paths).T.ffill(), last_weights_df

# ------------------------------------------------------------
# API ENDPUNKTE
# ------------------------------------------------------------

@app.route('/api/backtest', methods=['POST'])
def backtest():
    try:
        data = request.json
        settings = {
            'signal_accuracy': data.get('signal_accuracy', 0.8),
            'top_n_assets': data.get('top_n_assets', 3),
            'risk_target_ratio': data.get('risk_target_ratio', 0.9),
            'n_simulations': data.get('n_simulations', 20) # Reduziert für Speed
        }

        df_returns = load_data()
        paths_df, weights_history = run_simulation(df_returns, settings)
        
        # Berechnungen für Chart & Tabelle
        portfolio_median = paths_df.median(axis=1)
        portfolio_low = paths_df.quantile(0.1, axis=1)
        portfolio_high = paths_df.quantile(0.9, axis=1)
        
        bench_ret = df_returns[["equities", "bonds"]].mul(pd.Series(BENCHMARK_WEIGHTS), axis=1).sum(axis=1)
        bench_nav = (1 + bench_ret[bench_ret.index >= portfolio_median.index[0]]).cumprod()

        def get_stats(nav_series, label):
            total_ret = nav_series.pct_change().dropna()
            return {
                "strategy": label,
                "cagr": f"{((nav_series.iloc[-1]**(4/len(total_ret)))-1)*100:.1f} %",
                "vola": f"{total_ret.std() * math.sqrt(4) * 100:.1f} %",
                "max_dd": f"{(nav_series / nav_series.cummax() - 1).min() * 100:.1f} %",
                "sharpe": round((total_ret.mean() / total_ret.std()) * math.sqrt(4), 2) if total_ret.std() != 0 else 0
            }

        # HIER WERDEN BEIDE ZEILEN ERZEUGT
        summary = [
            get_stats(portfolio_median, "Portfolio Median"),
            get_stats(bench_nav, "Benchmark 70/30")
        ]

        return jsonify({
            "summary": summary,
            "chart": {
                "dates": [d.strftime('%Y-%m-%d') for d in portfolio_median.index],
                "portfolio_median": portfolio_median.tolist(),
                "portfolio_low": portfolio_low.tolist(),
                "portfolio_high": portfolio_high.tolist(),
                "benchmark": bench_nav.tolist()
            },
            "weights": weights_history
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
