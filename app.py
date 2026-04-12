from scenario_backtest_6assets import prepare_asset_returns, run_simulation_engine, DEFAULT_SETTINGS
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BacktestRequest(BaseModel):
    signal_accuracy: float = 0.8
    top_n_assets: int = 3
    risk_target_ratio: float = 0.9
    n_simulations: int = 100

@app.get("/")
def root():
    return {"status": "API läuft"}

@app.post("/api/backtest")
def run_backtest(req: BacktestRequest):
    try:
        settings = DEFAULT_SETTINGS.copy()
        settings["signal_accuracy"] = req.signal_accuracy
        settings["top_n_assets"] = req.top_n_assets
        settings["risk_target_ratio"] = req.risk_target_ratio
        settings["n_simulations"] = req.n_simulations
        settings["random_seed"] = 42
        settings["transaction_cost_bps"] = 10

        asset_returns = prepare_asset_returns()
        result = run_simulation_engine(asset_returns, settings)

        portfolio = result.portfolio_paths.median(axis=1)
        benchmark = result.benchmark_path.reindex(portfolio.index)

        return {
            "summary": result.summary.to_dict(orient="records"),
            "chart": {
                "dates": [str(d.date()) for d in portfolio.index],
                "portfolio_median": [None if pd.isna(x) else float(x) for x in portfolio],
                "benchmark": [None if pd.isna(x) else float(x) for x in benchmark]
            }
        }

    except Exception as e:
        return {
            "error": True,
            "detail": str(e)
        }
