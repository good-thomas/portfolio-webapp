#!/usr/bin/env python3
"""
scenario_backtest_6assets.py

Szenario-Backtest für ein 6-Asset-Portfolio mit quartalsweiser Umschichtung
auf Basis paarweiser Signale mit vorgegebener Trefferquote.

WICHTIG
- Das ist kein "echter" out-of-sample Backtest eines realen Signals
- Es ist eine Szenario-Simulation:
  Wir unterstellen, dass dein Signal paarweise in x% der Fälle die nächste Quartals-Performance
  korrekt ordnet, und simulieren daraus die Portfolioentwicklung
- Genau dafür ist das Skript gedacht

Benötigt:
    pip install yfinance pandas numpy matplotlib

Startbeispiel:
    python scenario_backtest_6assets.py

Optional mit Managed-Futures-CSV:
    python scenario_backtest_6assets.py --managed-futures-csv managed_futures.csv

CSV-Format für Managed Futures:
- Entweder:
    Date,Return
    2000-01-31,0.012
    2000-02-29,-0.004
  wobei Return als Dezimalzahl zu verstehen ist

- Oder:
    Date,Close
    2000-01-31,100.0
    2000-02-29,101.2

Ausgabe:
- performance_summary.csv
- portfolio_paths.csv
- quarterly_weights_last_run.csv
- backtest_chart.png
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
    raise SystemExit(
        "Bitte zuerst installieren: pip install yfinance pandas numpy matplotlib"
    )


# ------------------------------------------------------------
# Konfiguration
# ------------------------------------------------------------

START_DATE = "2000-01-01"
END_DATE = None

# Frei verfügbare Proxys
# Einige sind nicht perfekt, aber praktikabel für einen ersten Durchlauf
DEFAULT_TICKERS = {
    "equities": "SPY",          # Aktien
    "bonds": "VBMFX",           # US Aggregate Bond Fund Proxy
    "gold": "GC=F",             # Gold-Futures
    "commodities": "^SPGSCI",   # S&P GSCI
    "cash": "^IRX",             # 13-week T-Bill yield
    # managed_futures kommt optional über CSV
}

DEFAULT_SETTINGS = {
    "signal_accuracy": 0.80,    # 80% paarweise richtig
    "rebalance_frequency": "Q", # quartalsweise
    "top_n_assets": 3,          # wie viele Wachstumsmotoren gleichzeitig aktiv
    "risk_target_ratio": 0.90,  # Portfolio-Zielvola = 90% der Benchmark-Vola
    "max_asset_weight": 1.00,   # jede Assetklasse theoretisch 0 bis 100%
    "min_asset_weight": 0.00,
    "n_simulations": 300,
    "random_seed": 42,
    "transaction_cost_bps": 10,  # pro Quartal auf Turnover-Basis
}

BENCHMARK_WEIGHTS = {
    "equities": 0.70,
    "bonds": 0.30,
}


# ------------------------------------------------------------
# Datenstrukturen
# ------------------------------------------------------------

@dataclass
class BacktestResult:
    portfolio_paths: pd.DataFrame
    benchmark_path: pd.Series
    summary: pd.DataFrame
    weights_last_run: pd.DataFrame
    quarterly_asset_returns: pd.DataFrame


# ------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------

def max_drawdown(nav: pd.Series) -> float:
    running_max = nav.cummax()
    dd = nav / running_max - 1.0
    return float(dd.min())

def annualized_return(returns: pd.Series, periods_per_year: int = 4) -> float:
    n = len(returns)
    if n == 0:
        return np.nan
    total = (1 + returns).prod()
    return total ** (periods_per_year / n) - 1

def annualized_vol(returns: pd.Series, periods_per_year: int = 4) -> float:
    if len(returns) < 2:
        return np.nan
    return returns.std(ddof=1) * np.sqrt(periods_per_year)

def sharpe_ratio(returns: pd.Series, rf_per_period: float = 0.0, periods_per_year: int = 4) -> float:
    excess = returns - rf_per_period
    vol = excess.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return (excess.mean() / vol) * np.sqrt(periods_per_year)

def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()

def load_price_series_from_yf(ticker: str, start: str, end: str | None = None) -> pd.Series:
    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if data is None or data.empty:
        raise ValueError(f"Keine Daten für {ticker} geladen")

    # yfinance liefert teils MultiIndex, teils normal
    if isinstance(data.columns, pd.MultiIndex):
        if ("Close", ticker) in data.columns:
            s = data[("Close", ticker)].copy()
        elif ("Adj Close", ticker) in data.columns:
            s = data[("Adj Close", ticker)].copy()
        else:
            s = data.iloc[:, 0].copy()
    else:
        if "Close" in data.columns:
            s = data["Close"].copy()
        elif "Adj Close" in data.columns:
            s = data["Adj Close"].copy()
        else:
            s = data.iloc[:, 0].copy()

    s.name = ticker
    return s.dropna()

def load_managed_futures_csv(path: str | Path) -> pd.Series:
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError("Managed-Futures-CSV braucht eine Spalte 'Date'")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    if "Return" in df.columns:
        ret = df["Return"].astype(float).copy()
        nav = (1 + ret).cumprod()
        nav.name = "managed_futures"
        return nav

    if "Close" in df.columns:
        nav = df["Close"].astype(float).copy()
        nav.name = "managed_futures"
        return nav

    raise ValueError("CSV braucht entweder Spalte 'Return' oder 'Close'")

def quarterly_returns_from_price(price: pd.Series) -> pd.Series:
    q = price.resample("QE").last().dropna()
    return q.pct_change().dropna()

def quarterly_cash_returns_from_irx(irx_yield: pd.Series) -> pd.Series:
    # ^IRX ist annualisierte T-Bill-Yield in Prozent
    q = irx_yield.resample("QE").last().dropna()
    ret = (q / 100.0) / 4.0
    ret.name = "cash"
    return ret.dropna()

def prepare_asset_returns(managed_futures_csv: str | None = None) -> pd.DataFrame:
    returns = {}

    for asset, ticker in DEFAULT_TICKERS.items():
        s = load_price_series_from_yf(ticker, START_DATE, END_DATE)
        if asset == "cash":
            returns[asset] = quarterly_cash_returns_from_irx(s)
        else:
            returns[asset] = quarterly_returns_from_price(s)

    if managed_futures_csv:
        mf_nav = load_managed_futures_csv(managed_futures_csv)
        returns["managed_futures"] = quarterly_returns_from_price(mf_nav)
else:
    print("INFO: Verwende DBMF als Managed-Futures-Proxy")

    mf_price = load_price_series_from_yf("DBMF", START_DATE, END_DATE)
    returns["managed_futures"] = quarterly_returns_from_price(mf_price)
        )

    df = pd.concat(returns, axis=1)
    df = df.dropna(how="any")

    # Reihenfolge möglichst nah an deiner Vorgabe
    desired_order = [
        "equities",
        "bonds",
        "managed_futures",
        "gold",
        "commodities",
        "cash",
    ]
    cols = [c for c in desired_order if c in df.columns]
    return df[cols].copy()

def benchmark_returns(asset_returns: pd.DataFrame) -> pd.Series:
    w = pd.Series(BENCHMARK_WEIGHTS)
    common = [c for c in w.index if c in asset_returns.columns]
    bench = asset_returns[common].mul(w[common], axis=1).sum(axis=1)
    bench.name = "benchmark_70_30"
    return bench

def infer_pairwise_scores(next_returns: pd.Series, accuracy: float, rng: np.random.Generator) -> pd.Series:
    """
    Baut aus den tatsächlich folgenden Quartalsrenditen ein künstliches Signal:
    Für jedes Assetpaar entscheidet das Signal mit Wahrscheinlichkeit 'accuracy'
    korrekt, welches Asset im nächsten Quartal besser läuft.
    Daraus wird per Borda-Count ein Ranking abgeleitet.
    """
    assets = list(next_returns.index)
    wins = pd.Series(0.0, index=assets)

    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            a, b = assets[i], assets[j]
            ra, rb = next_returns[a], next_returns[b]

            if np.isclose(ra, rb):
                # Gleichstand -> halber Punkt
                wins[a] += 0.5
                wins[b] += 0.5
                continue

            correct_winner = a if ra > rb else b
            correct = rng.random() < accuracy
            predicted_winner = correct_winner if correct else (b if correct_winner == a else a)

            wins[predicted_winner] += 1.0

    return wins.sort_values(ascending=False)

def build_target_weights(
    signal_scores: pd.Series,
    trailing_vol: pd.Series,
    top_n_assets: int,
    min_asset_weight: float,
    max_asset_weight: float,
) -> pd.Series:
    """
    Aus den paarweisen Scores wird ein Zielgewicht abgeleitet:
    - nur die Top-N Assets werden als Wachstumsmotoren genutzt
    - innerhalb der Top-N erfolgt Gewichtung score-basiert
    - zusätzlich inverse Volatilität, damit das Risiko nicht komplett klumpt
    """
    assets = signal_scores.index.tolist()
    selected = signal_scores.nlargest(min(top_n_assets, len(signal_scores))).index.tolist()

    raw = pd.Series(0.0, index=assets)

    selected_scores = signal_scores[selected].astype(float)
    selected_scores = selected_scores - selected_scores.min() + 1.0

    selected_vol = trailing_vol[selected].replace(0, np.nan).fillna(trailing_vol.median())
    inv_vol = 1.0 / selected_vol

    combo = selected_scores * inv_vol
    combo = combo / combo.sum()

    raw.loc[selected] = combo.values

    # Hart auf min/max beschneiden
    raw = raw.clip(lower=min_asset_weight, upper=max_asset_weight)

    # Auf 1 normieren; falls leer -> 100% Cash
    if raw.sum() <= 0:
        if "cash" in raw.index:
            raw.loc[:] = 0.0
            raw["cash"] = 1.0
        else:
            raw.loc[:] = 1.0 / len(raw)
    else:
        raw = raw / raw.sum()

    return raw

def apply_risk_overlay(
    target_weights: pd.Series,
    trailing_cov: pd.DataFrame,
    benchmark_vol_q: float,
    risk_target_ratio: float,
) -> pd.Series:
    """
    Hebt oder senkt das Risiko des Zielportfolios relativ zur Benchmark.
    Rest geht in Cash.
    """
    assets = target_weights.index.tolist()

    risky_assets = [a for a in assets if a != "cash"]
    w_risky = target_weights[risky_assets].copy()

    if w_risky.sum() <= 0:
        out = target_weights.copy()
        if "cash" in out.index:
            out.loc[:] = 0.0
            out["cash"] = 1.0
        return out

    cov = trailing_cov.loc[risky_assets, risky_assets].copy()
    w_vec = w_risky.values.reshape(-1, 1)
    port_var = float((w_vec.T @ cov.values @ w_vec).squeeze())
    port_vol_q = math.sqrt(max(port_var, 0.0))

    target_vol_q = benchmark_vol_q * risk_target_ratio

    if port_vol_q <= 0 or np.isnan(port_vol_q):
        leverage = 1.0
    else:
        leverage = min(1.0, target_vol_q / port_vol_q)

    scaled = target_weights.copy() * leverage

    if "cash" in scaled.index:
        scaled["cash"] += 1.0 - scaled.sum()
    else:
        scaled = scaled / scaled.sum()

    scaled = scaled.clip(lower=0.0)
    scaled = scaled / scaled.sum()
    return scaled

def run_one_simulation(
    asset_returns: pd.DataFrame,
    bench_returns: pd.Series,
    settings: dict,
    rng: np.random.Generator,
) -> tuple[pd.Series, pd.DataFrame]:
    dates = asset_returns.index
    assets = asset_returns.columns.tolist()

    # initial gleichgewichtet
    current_weights = pd.Series(1.0 / len(assets), index=assets)
    nav = [1.0]
    nav_index = [dates[0] - pd.offsets.QuarterEnd(1)]
    weight_records = []

    trailing_window = 8  # 8 Quartale = 2 Jahre

    for t in range(trailing_window, len(dates) - 1):
        date_t = dates[t]
        next_date = dates[t + 1]

        trailing = asset_returns.iloc[t - trailing_window:t]
        next_q = asset_returns.iloc[t + 1]

        trailing_vol = trailing.std(ddof=1).replace(0, np.nan).fillna(trailing.std(ddof=1).median())
        trailing_cov = trailing.cov()

        bench_vol_q = trailing[["equities", "bonds"]].mul(
            pd.Series(BENCHMARK_WEIGHTS), axis=1
        ).sum(axis=1).std(ddof=1)

        signal_scores = infer_pairwise_scores(
            next_returns=next_q,
            accuracy=settings["signal_accuracy"],
            rng=rng,
        )

        target_weights = build_target_weights(
            signal_scores=signal_scores,
            trailing_vol=trailing_vol,
            top_n_assets=settings["top_n_assets"],
            min_asset_weight=settings["min_asset_weight"],
            max_asset_weight=settings["max_asset_weight"],
        )

        target_weights = apply_risk_overlay(
            target_weights=target_weights,
            trailing_cov=trailing_cov,
            benchmark_vol_q=bench_vol_q,
            risk_target_ratio=settings["risk_target_ratio"],
        )

        turnover = float((target_weights - current_weights).abs().sum())
        tc = turnover * settings["transaction_cost_bps"] / 10000.0

        realized_ret = float((target_weights * next_q).sum() - tc)
        nav.append(nav[-1] * (1 + realized_ret))
        nav_index.append(next_date)

        rec = target_weights.copy()
        rec.name = next_date
        weight_records.append(rec)

        current_weights = target_weights.copy()

    nav_series = pd.Series(nav, index=pd.DatetimeIndex(nav_index), name="portfolio_nav")
    weights_df = pd.DataFrame(weight_records)
    return nav_series, weights_df

def run_simulation_engine(asset_returns: pd.DataFrame, settings: dict) -> BacktestResult:
    rng = np.random.default_rng(settings["random_seed"])
    bench_ret = benchmark_returns(asset_returns)

    sims = {}
    last_weights = None

    for i in range(settings["n_simulations"]):
        path, weights = run_one_simulation(asset_returns, bench_ret, settings, rng)
        sims[f"sim_{i+1}"] = path
        last_weights = weights

    portfolio_paths = pd.concat(sims, axis=1).ffill()

    # Benchmark auf denselben Zeitraum zuschneiden
    start = portfolio_paths.index.min()
    aligned_bench_ret = bench_ret[bench_ret.index >= start]
    bench_nav = (1 + aligned_bench_ret).cumprod()
    bench_nav = bench_nav / bench_nav.iloc[0]
    bench_nav.name = "benchmark_70_30"

    # Kennzahlen
    summary_rows = []

    # Benchmark
    summary_rows.append(
        {
            "strategy": "benchmark_70_30",
            "cagr": annualized_return(aligned_bench_ret),
            "ann_vol": annualized_vol(aligned_bench_ret),
            "max_drawdown": max_drawdown(bench_nav),
            "sharpe": sharpe_ratio(aligned_bench_ret),
            "ending_nav": float(bench_nav.iloc[-1]),
        }
    )

    # Simulationsverteilung
    sim_stats = []
    for col in portfolio_paths.columns:
        nav = portfolio_paths[col].dropna()
        rets = nav.pct_change().dropna()
        sim_stats.append(
            {
                "strategy": col,
                "cagr": annualized_return(rets),
                "ann_vol": annualized_vol(rets),
                "max_drawdown": max_drawdown(nav),
                "sharpe": sharpe_ratio(rets),
                "ending_nav": float(nav.iloc[-1]),
            }
        )

    sim_df = pd.DataFrame(sim_stats)

    summary_rows.extend(
        [
            {
                "strategy": "portfolio_median",
                "cagr": sim_df["cagr"].median(),
                "ann_vol": sim_df["ann_vol"].median(),
                "max_drawdown": sim_df["max_drawdown"].median(),
                "sharpe": sim_df["sharpe"].median(),
                "ending_nav": sim_df["ending_nav"].median(),
            },
            {
                "strategy": "portfolio_p10",
                "cagr": sim_df["cagr"].quantile(0.10),
                "ann_vol": sim_df["ann_vol"].quantile(0.10),
                "max_drawdown": sim_df["max_drawdown"].quantile(0.10),
                "sharpe": sim_df["sharpe"].quantile(0.10),
                "ending_nav": sim_df["ending_nav"].quantile(0.10),
            },
            {
                "strategy": "portfolio_p90",
                "cagr": sim_df["cagr"].quantile(0.90),
                "ann_vol": sim_df["ann_vol"].quantile(0.90),
                "max_drawdown": sim_df["max_drawdown"].quantile(0.90),
                "sharpe": sim_df["sharpe"].quantile(0.90),
                "ending_nav": sim_df["ending_nav"].quantile(0.90),
            },
        ]
    )

    summary = pd.DataFrame(summary_rows)
    return BacktestResult(
        portfolio_paths=portfolio_paths,
        benchmark_path=bench_nav,
        summary=summary,
        weights_last_run=last_weights,
        quarterly_asset_returns=asset_returns,
    )

def make_plot(result: BacktestResult, out_png: str = "backtest_chart.png") -> None:
    plt.figure(figsize=(12, 7))

    # Unsicherheitsband der Simulationen
    common_index = result.portfolio_paths.index
    paths = result.portfolio_paths.reindex(common_index).ffill()
    q10 = paths.quantile(0.10, axis=1)
    q50 = paths.quantile(0.50, axis=1)
    q90 = paths.quantile(0.90, axis=1)

    plt.fill_between(common_index, q10, q90, alpha=0.20, label="Portfolio 10%-90% Band")
    plt.plot(common_index, q50, linewidth=2.2, label="Portfolio Median")
    plt.plot(result.benchmark_path.index, result.benchmark_path.values, linewidth=2.0, label="Benchmark 70/30")

    plt.yscale("log")
    plt.title("6-Asset-Szenario vs. 70/30 seit 2000")
    plt.ylabel("NAV (log-Skala)")
    plt.xlabel("Datum")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def save_outputs(result: BacktestResult, output_dir: str | Path = ".") -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result.summary.to_csv(output_dir / "performance_summary.csv", index=False)
    result.portfolio_paths.to_csv(output_dir / "portfolio_paths.csv")
    result.weights_last_run.to_csv(output_dir / "quarterly_weights_last_run.csv")
    result.quarterly_asset_returns.to_csv(output_dir / "quarterly_asset_returns.csv")
    make_plot(result, str(output_dir / "backtest_chart.png"))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--managed-futures-csv", type=str, default=None)
    p.add_argument("--signal-accuracy", type=float, default=DEFAULT_SETTINGS["signal_accuracy"])
    p.add_argument("--top-n-assets", type=int, default=DEFAULT_SETTINGS["top_n_assets"])
    p.add_argument("--risk-target-ratio", type=float, default=DEFAULT_SETTINGS["risk_target_ratio"])
    p.add_argument("--n-simulations", type=int, default=DEFAULT_SETTINGS["n_simulations"])
    p.add_argument("--transaction-cost-bps", type=float, default=DEFAULT_SETTINGS["transaction_cost_bps"])
    p.add_argument("--output-dir", type=str, default="backtest_output")
    return p.parse_args()

def main():
    args = parse_args()

    settings = DEFAULT_SETTINGS.copy()
    settings["signal_accuracy"] = args.signal_accuracy
    settings["top_n_assets"] = args.top_n_assets
    settings["risk_target_ratio"] = args.risk_target_ratio
    settings["n_simulations"] = args.n_simulations
    settings["transaction_cost_bps"] = args.transaction_cost_bps

    asset_ret = prepare_asset_returns(managed_futures_csv=args.managed_futures_csv)
    result = run_simulation_engine(asset_ret, settings=settings)
    save_outputs(result, args.output_dir)

    print("\nFertig. Wichtige Dateien:")
    print(f"  {args.output_dir}/performance_summary.csv")
    print(f"  {args.output_dir}/portfolio_paths.csv")
    print(f"  {args.output_dir}/quarterly_weights_last_run.csv")
    print(f"  {args.output_dir}/quarterly_asset_returns.csv")
    print(f"  {args.output_dir}/backtest_chart.png")
    print("\nKennzahlen:")
    print(result.summary.to_string(index=False))

if __name__ == "__main__":
    main()
