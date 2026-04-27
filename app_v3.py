import pandas as pd
import numpy as np
import yfinance as yf
import math

# --- KONFIGURATION ---
START_DATE = "2015-01-01"  # Für Defense-Historie (DFND) oft passender Start
COST_RATE = 0.001          # 0,1% Transaktionskosten

# 1. Assetklassen-Universum (Neutrale Basis)
NEUTRAL_WEIGHTS = {
    "equities": 0.50,
    "bonds": 0.20,
    "managed_futures": 0.10,
    "gold": 0.10,
    "bitcoin": 0.05,
    "cash": 0.05
}

TICKERS = {
    "equities": "ACWI",        # MSCI World Basis
    "bonds": "IEF",            # 7-10y US Treasuries
    "gold": "GLD",             # Gold Spot
    "managed_futures": "DBMF", # Managed Futures Proxy
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

def compute_score(series):
    """Deine Formel: 3M * 0.5 + 6M * 0.3 + 12M * 0.2"""
    if len(series) < 13: return -999
    r3 = series.iloc[-1] / series.iloc[-4] - 1
    r6 = series.iloc[-1] / series.iloc[-7] - 1
    r12 = series.iloc[-1] / series.iloc[-13] - 1
    return (r3 * 0.5) + (r6 * 0.3) + (r12 * 0.2)

def run_ambitious_backtest():
    # 1. Daten laden
    all_tickers = list(TICKERS.values()) + list(GICS_BUCKETS.values())
    data = yf.download(all_tickers, start=START_DATE, auto_adjust=True)['Close']
    prices = data.resample('ME').last().ffill()
    returns = prices.pct_change().dropna()

    # 2. Backtest Variablen
    nav = [1.0]
    bench_nav = [1.0]
    dates = []
    prev_weights = pd.Series(0.0, index=all_tickers)

    # Asset Namen Mapping für leichteren Zugriff
    t_map = {v: k for k, v in TICKERS.items()}
    g_map = {v: k for k, v in GICS_BUCKETS.items()}

    for i in range(13, len(prices) - 1):
        current_date = prices.index[i]
        dates.append(prices.index[i+1])
        
        # --- EBENE 1: MULTI-ASSET RANKING ---
        asset_scores = {}
        for name, ticker in TICKERS.items():
            if name != "cash":
                asset_scores[name] = compute_score(prices[ticker].iloc[:i+1])
        
        # Ranking und Top 3
        ranking = sorted(asset_scores, key=asset_scores.get, reverse=True)
        top3 = ranking[:3]
        
        # Gewichte mit Winner-Zuschlag
        # Platz 1: +10%, Platz 2: +5%, Platz 3: +2%
        w_step = NEUTRAL_WEIGHTS.copy()
        boosts = {0: 0.10, 1: 0.05, 2: 0.02}
        
        total_boost = 0
        for rank, asset in enumerate(top3):
            # Nur investieren, wenn Score > 0 (Absolute Momentum Filter)
            if asset_scores[asset] > 0:
                w_step[asset] += boosts[rank]
                total_boost += boosts[rank]
            else:
                # Fallback: Boost geht in Cash
                w_step["cash"] += boosts[rank]
                total_boost += boosts[rank]

        # Um 100% zu halten, kürzen wir Bonds & Equities Core anteilig
        # (Einfache Lösung für dieses Script)
        w_step["equities"] -= (total_boost * 0.7)
        w_step["bonds"] -= (total_boost * 0.3)

        # --- EBENE 2: GICS BUCKET ALPHA ---
        # Wir nehmen 15% aus der Aktienquote (50%) und setzen sie in Top 3 Buckets
        bucket_scores = {}
        for b_name, b_ticker in GICS_BUCKETS.items():
            bucket_scores[b_ticker] = compute_score(prices[b_ticker].iloc[:i+1])
        
        top_buckets = sorted(bucket_scores, key=bucket_scores.get, reverse=True)[:3]
        
        # Finales Weighting-Objekt für diesen Monat
        current_weights = pd.Series(0.0, index=all_tickers)
        
        # Multi-Asset Zuweisung
        for name, weight in w_step.items():
            if name == "equities":
                # 35% MSCI World (Beta), 15% Alpha Buckets
                current_weights[TICKERS["equities"]] = weight - 0.15
                for tb in top_buckets:
                    current_weights[tb] = 0.05 # 3 x 5% = 15%
            else:
                current_weights[TICKERS[name]] = weight

        # --- PERFORMANCE BERECHNUNG ---
        r_next = returns.iloc[i] # Performance des nächsten Monats
        
        # Portfolio Performance inkl. Kosten
        turnover = (current_weights - prev_weights).abs().sum()
        costs = turnover * COST_RATE
        port_return = (current_weights * r_next).sum() - costs
        nav.append(nav[-1] * (1.0 + port_return))
        
        # Benchmark Performance (70/30 MSCI World / IEF)
        bench_return = 0.70 * r_next[TICKERS["equities"]] + 0.30 * r_next[TICKERS["bonds"]]
        bench_nav.append(bench_nav[-1] * (1.0 + bench_return))
        
        prev_weights = current_weights

    # --- STATISTIKEN ---
    results = pd.DataFrame({"Portfolio": nav, "Benchmark": bench_nav}, index=[prices.index[13]] + dates)
    
    def get_stats(series):
        rets = series.pct_change().dropna()
        cagr = (series.iloc[-1]** (1/(len(series)/12))) - 1
        vol = rets.std() * math.sqrt(12)
        dd = (series / series.cummax() - 1).min()
        return f"CAGR: {cagr:.1%}, Vol: {vol:.1%}, MaxDD: {dd:.1%}, Sharpe: {cagr/vol:.2f}"

    print(f"Strategie: {get_stats(results['Portfolio'])}")
    print(f"Benchmark: {get_stats(results['Benchmark'])}")
    
    return results

if __name__ == "__main__":
    results = run_ambitious_backtest()
    # results.plot(title="Ambitioniertes Portfolio vs. 70/30 Benchmark")
