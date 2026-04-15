#!/usr/bin/env python3
"""
scenario_backtest_7assets_crypto.py
Angepasste Version für Thomas: Inklusive Krypto und robuster Zeitreihen-Logik.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Bitte installieren: pip install yfinance pandas numpy matplotlib")

# ------------------------------------------------------------
# Konfiguration
# ------------------------------------------------------------

START_DATE = "2000-01-01"
END_DATE = None

DEFAULT_TICKERS = {
    "equities": "SPY",
    "bonds": "VBMFX",
    "gold": "GC=F",
    "commodities": "^SPGSCI",
    "cash": "^IRX",
    "crypto": "BTC-USD"  # Bitcoin kommt dazu
}

DEFAULT_SETTINGS = {
    "signal_accuracy": 0.80,
    "rebalance_frequency": "Q",
    "top_n_assets": 3,
    "risk_target_ratio": 0.90,
    "max_asset_weight": 1.00,
    "min_asset_weight": 0.00,
    "n_simulations": 300,
    "random_seed": 42,
    "transaction_cost_bps": 10,
}

BENCHMARK_WEIGHTS = {"equities": 0.70, "bonds": 0.30}

# ------------------------------------------------------------
# Hilfsfunktionen & Kern-Logik
# ------------------------------------------------------------

def max_drawdown(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1.0).min())

def annualized_return(returns: pd.Series, periods_per_year: int = 4) -> float:
    if len(returns) == 0: return np.nan
    return (1 + returns).prod() ** (periods_per_year / len(returns)) - 1

def annualized_vol(returns: pd.Series, periods_per_year: int = 4) -> float:
    if len(returns) < 2: return np.nan
    return returns.std(ddof=1) * np.sqrt(periods_per_year)

def sharpe_ratio(returns: pd.Series, rf_per_period: float = 0.0, periods_per_year: int = 4) -> float:
    excess = returns - rf_per_period
    vol = excess.std(ddof=1)
    if vol == 0 or np.isnan(vol): return np.nan
    return (excess.mean() / vol) * np.sqrt(periods_per_year)

def load_price_series_from_yf(ticker: str, start: str, end: str | None = None) -> pd.Series:
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if data is None or data.empty:
        print(f"Hinweis: Keine Daten für {ticker}")
        return pd.Series(dtype=float)
    
    # Sicherstellen, dass wir eine Series extrahieren (bei MultiIndex durch yf)
    if isinstance(data.columns, pd.MultiIndex):
        s = data.iloc[:, 0].copy()
    else:
        s = data["Close"].copy() if "Close" in data.columns else data.iloc[:, 0].copy()
    
    s.name = ticker
    return s.dropna()

def prepare_asset_returns(managed_futures_csv: str | None = None) -> pd.DataFrame:
    returns = {}
    for asset, ticker in DEFAULT_TICKERS.items():
        s = load_price_series_from_yf(ticker, START_DATE, END_DATE)
        if s.empty: continue
        
        if asset == "cash":
            # ^IRX ist annualisierte Yield
            q = s.resample("QE").last()
            ret = (q / 100.0) / 4.0
            returns[asset] = ret
        else:
            returns[asset] = s.resample("QE").last().pct_change()

    if managed_futures_csv:
        df_mf = pd.read_csv(managed_futures_csv)
        df_mf["Date"] = pd.to_datetime(df_mf["Date"])
        df_mf = df_mf.set_index("Date").sort_index()
        nav = df_mf["Close"] if "Close" in df_mf.columns else (1 + df_mf["Return"]).cumprod()
        returns["managed_futures"] = nav.resample("QE").last().pct_change()

    df = pd.concat(returns, axis=1)
    # WICHTIG: Nicht alles löschen (dropna), sondern mit 0 auffüllen für Assets, die später starten
    df = df.fillna(0.0) 
    return df.iloc[1:] # Erste Zeile ist meist NaN vom pct_change

def infer_pairwise_scores(next_returns: pd.Series, accuracy: float, rng: np.random.Generator, valid_assets: list) -> pd.Series:
    assets = valid_assets
    wins = pd.Series(0.0, index=assets)
    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            a, b = assets[i], assets[j]
            ra, rb = next_returns[a], next_returns[b]
            if np.isclose(ra, rb):
                wins[a] += 0.5; wins[b] += 0.5
                continue
            correct_winner = a if ra > rb else b
            predicted_winner = correct_winner if rng.random() < accuracy else (b if correct_winner == a else a)
            wins[predicted_winner] += 1.0
    return wins.sort_values(ascending=False)

def build_target_weights(signal_scores: pd.Series, trailing_vol: pd.Series, top_n_assets: int, min_weight: float, max_weight: float) -> pd.Series:
    # Nur Assets, die im Zeitraum existiert haben (Vola > 0)
    valid = trailing_vol[trailing_vol > 0.00001].index
    scores = signal_scores.reindex(valid).dropna()
    
    selected = scores.nlargest(min(top_n_assets, len(scores))).index
    raw = pd.Series(0.0, index=signal_scores.index)
    
    if len(selected) > 0:
        inv_vol = 1.0 / trailing_vol[selected].replace(0, 0.01)
        weights = (scores[selected] + 1) * inv_vol
        raw.loc[selected] = weights / weights.sum()

    raw = raw.clip(lower=min_weight, upper=max_weight)
    return raw / raw.sum() if raw.sum() > 0 else raw

def apply_risk_overlay(target_weights: pd.Series, trailing_cov: pd.DataFrame, bench_vol: float, ratio: float) -> pd.Series:
    risky = [a for a in target_weights.index if a != "cash" and target_weights[a] > 0]
    if not risky or bench_vol == 0: return target_weights
    
    w_vec = target_weights[risky].values
    cov = trailing_cov.loc[risky, risky].values
    port_vol = np.sqrt(w_vec.T @ cov @ w_vec)
    
    leverage = min(1.0, (bench_vol * ratio) / port_vol) if port_vol > 0 else 1.0
    scaled = target_weights * leverage
    if "cash" in scaled.index: scaled["cash"] += 1.0 - scaled.sum()
    return scaled.clip(lower=0)

def run_one_simulation(asset_returns: pd.DataFrame, settings: dict, rng: np.random.Generator) -> tuple[pd.Series, pd.DataFrame]:
    dates = asset_returns.index
    assets = asset_returns.columns.tolist()
    current_weights = pd.Series(1.0 / len(assets), index=assets)
    nav, nav_index, weight_records = [1.0], [dates[0]], []
    
    window = 8
    for t in range(window, len(dates) - 1):
        trailing = asset_returns.iloc[t-window:t]
        next_q = asset_returns.iloc[t+1]
        
        # Nur Assets mit Aktivität im Fenster
        trailing_vol = trailing.std(ddof=1).fillna(0)
        valid_now = trailing_vol[trailing_vol > 0.00001].index.tolist()
        
        bench_vol = trailing[["equities", "bonds"]].mul(pd.Series(BENCHMARK_WEIGHTS), axis=1).sum(axis=1).std(ddof=1)
        
        scores = infer_pairwise_scores(next_q, settings["signal_accuracy"], rng, valid_now)
        weights = build_target_weights(scores, trailing_vol, settings["top_n_assets"], settings["min_asset_weight"], settings["max_asset_weight"])
        weights = apply_risk_overlay(weights, trailing.cov().fillna(0), bench_vol, settings["risk_target_ratio"])
        
        tc = (weights - current_weights).abs().sum() * settings["transaction_cost_bps"] / 10000.0
        realized_ret = (weights * next_q).sum() - tc
        nav.append(nav[-1] * (1 + realized_ret))
        nav_index.append(dates[t+1])
        weight_records.append(weights.rename(dates[t+1]))
        current_weights = weights

    return pd.Series(nav, index=pd.DatetimeIndex(nav_index)), pd.DataFrame(weight_records)

# ------------------------------------------------------------
# Engine & Output (verkürzt für Übersicht)
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--managed-futures-csv", type=str, default=None)
    parser.add_argument("--n-simulations", type=int, default=100)
    args = parser.parse_args()

    asset_ret = prepare_asset_returns(args.managed_futures_csv)
    bench_ret = asset_ret[["equities", "bonds"]].mul(pd.Series(BENCHMARK_WEIGHTS), axis=1).sum(axis=1)
    
    rng = np.random.default_rng(42)
    sim_paths = {}
    
    print(f"Starte {args.n_simulations} Simulationen über {len(asset_ret)} Quartale...")
    for i in range(args.n_simulations):
        path, _ = run_one_simulation(asset_ret, DEFAULT_SETTINGS, rng)
        sim_paths[f"sim_{i}"] = path

    # Auswertung & Plotting
    paths_df = pd.DataFrame(sim_paths).ffill()
    bench_nav = (1 + bench_ret[bench_ret.index >= paths_df.index[0]]).cumprod()
    
    plt.figure(figsize=(10, 6))
    plt.plot(paths_df.median(axis=1), label="Portfolio Median", color="blue")
    plt.plot(bench_nav / bench_nav[0], label="Benchmark 70/30", color="red", linestyle="--")
    plt.yscale("log")
    plt.title("Backtest: 7 Assets inkl. Krypto (Start 2000)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("backtest_krypto.png")
    print("Backtest abgeschlossen. Chart gespeichert als 'backtest_krypto.png'.")

if __name__ == "__main__":
    main()
