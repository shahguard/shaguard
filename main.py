import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import yfinance as yf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

market_data_global = None
oracle = None # Initialize oracle as None globally

class OptimizeRoutingRequest(BaseModel):
    timestamp_idx: int
    amount: float

class ComputeSHARequest(BaseModel):
    timestamp_idx: int
    amount: float

class RoutingResponse(BaseModel):
    optimal_routing: List[float]
    liquidity_depths: List[float]
    chains: List[str]
    amount: float

class SHAResponse(BaseModel):
    sha_metric: float
    liquidity_depths: List[float]
    chains: List[str]
    amount: float

def load_market_data_for_api():
    logging.info("📥 Loading real market data for API...")
    symbols = ["ETH-USD", "BTC-USD", "SOL-USD", "ARB-USD", "OP-USD"]
    crypto_data = {}
    for symbol in symbols:
        csv_file = symbol.replace('-USD', '.csv')
        file_path = f'/content/{csv_file}'
        try:
            data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            if not data.empty and len(data) > 10 and 'Close' in data.columns and 'Volume' in data.columns:
                data = data.sort_index()
                crypto_data[symbol] = data
                logging.info(f"  ✓ {symbol}: {len(data)} daily data points loaded from {file_path}")
            else:
                raise ValueError("Insufficient data or missing columns")
        except FileNotFoundError:
            logging.warning(f"  ✗ {symbol}: CSV file not found at {file_path}, using synthetic data")
            n_days = 365
            dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
            close = 3000 + np.cumsum(np.random.randn(n_days) * 50)
            volume = np.random.lognormal(15, 1, n_days)
            crypto_data[symbol] = pd.DataFrame({
                'Close': close,
                'Volume': volume
            }, index=dates)
        except Exception as e:
            logging.error(f"  ✗ {symbol}: Error loading CSV, using synthetic data: {str(e)}")
            n_days = 365
            dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
            close = 3000 + np.cumsum(np.random.randn(n_days) * 50)
            volume = np.random.lognormal(15, 1, n_days)
            crypto_data[symbol] = pd.DataFrame({
                'Close': close,
                'Volume': volume
            }, index=dates)

    logging.info("\nℹ Generating realistic liquidity profiles...")
    chains = ['ethereum', 'arbitrum', 'optimism', 'polygon', 'base']
    base_liquidity = {
        'ethereum': 2.5e9,
        'arbitrum': 1.8e9,
        'optimism': 8e8,
        'polygon': 1.2e9,
        'base': 6e8
    }
    if 'ETH-USD' in crypto_data and not crypto_data['ETH-USD'].empty:
        end_date = crypto_data['ETH-USD'].index.max()
        start_date = end_date - timedelta(days=90)
    else:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)

    timestamps = pd.date_range(start=start_date, end=end_date, freq='6H')
    liquidity_data = []
    for ts in timestamps:
        row = {'time': ts}
        for chain in chains:
            symbol = 'ETH-USD' if chain in ['ethereum', 'base'] else \
                    'ARB-USD' if chain == 'arbitrum' else \
                    'OP-USD' if chain == 'optimism' else 'MATIC-USD'
            data = crypto_data.get(symbol, pd.DataFrame())
            price_factor = volume_factor = volatility = 1.0
            if not data.empty:
                try:
                    closest_date_data = data.asof(ts.normalize())
                    if closest_date_data is not None and not closest_date_data.empty:
                        close_series = data['Close'].dropna()
                        volume_series = data['Volume'].dropna()
                        if not close_series.empty and close_series.iloc[0] != 0 and pd.api.types.is_numeric_dtype(close_series):
                             price_factor = closest_date_data['Close'] / close_series.iloc[0]
                        else:
                            price_factor = 1.0
                        if not volume_series.empty and volume_series.iloc[0] != 0 and pd.api.types.is_numeric_dtype(volume_series):
                             volume_factor = np.log(closest_date_data['Volume'] / volume_series.iloc[0] + 1) + 1
                        else:
                            volume_factor = 1.0
                        if len(close_series) > 7:
                            current_data_date = closest_date_data.name.normalize()
                            vol_start_date = current_data_date - timedelta(days=7)
                            recent_data_for_vol = close_series[(close_series.index.normalize() >= vol_start_date) & (close_series.index.normalize() <= current_data_date)]
                            pct_change = recent_data_for_vol.pct_change().dropna()
                            volatility = pct_change.std() if not pct_change.empty else 0.02
                        else:
                            volatility = 0.02
                    else:
                        price_factor = 1.0
                        volume_factor = 1.0
                        volatility = 1.0
                except Exception as e:
                    logging.error(f"Error processing {symbol} at {ts.date()}: {str(e)}")
                    price_factor = 1.0
                    volume_factor = 1.0
                    volatility = 1.0
            volatility_factor = max(0.2, 1 - volatility * 5)
            base = base_liquidity[chain]
            liquidity = float(base * price_factor * volume_factor * volatility_factor * (1 + np.random.randn() * 0.3))
            row[chain] = max(liquidity, base * 0.1)
        liquidity_data.append(row)
    eth_liquidity = pd.DataFrame(liquidity_data)

    mev_data = []
    for ts in timestamps[::24]:
        mev_profit = float(np.random.lognormal(12, 1))
        mev_data.append({'block_time': ts.date(), 'profit': mev_profit})
    mev_df = pd.DataFrame(mev_data)

    logging.info(f"\n✅ Data preparation complete!")
    logging.info(f"   • Time range: {start_date.date()} to {end_date.date()}")
    logging.info(f"   • Data points: {len(eth_liquidity)}")
    logging.info(f"   • Chains: {chains}")

    return {
        'eth_liquidity': eth_liquidity,
        'crypto_data': crypto_data,
        'mev_data': mev_df,
        'chains': chains
    }

class ShaOracle:
    """The Cohomological Liquidity Router - Production Implementation"""

    def __init__(self, market_data):
        self.market_data = market_data
        self.chains = market_data['chains']
        logging.info("✅ Ш-Oracle Protocol initialized successfully")

    def get_volatility_factors(self, timestamp_idx):
        """Calculate volatility factors for each chain"""
        volatility = []
        for chain in self.chains:
            symbol = 'ETH-USD' if chain in ['ethereum', 'base'] else \
                    'ARB-USD' if chain == 'arbitrum' else \
                    'OP-USD' if chain == 'optimism' else 'MATIC-USD'
            data = self.market_data['crypto_data'].get(symbol, pd.DataFrame())
            if not data.empty and len(data) > 7:
                try:
                    eth_liquidity_time = pd.Timestamp(self.market_data['eth_liquidity'].iloc[timestamp_idx]['time'])

                    closest_date_data = data.asof(eth_liquidity_time.normalize()) # Normalize for consistent Timestamp lookup

                    if closest_date_data is not None and not closest_date_data.empty:
                         close_series = data['Close'].dropna()

                         if len(close_series) > 7:
                            current_data_date = closest_date_data.name.normalize()
                            vol_start_date = current_data_date - timedelta(days=7)
                            recent_data_for_vol = close_series[(close_series.index.normalize() >= vol_start_date) & (close_series.index.normalize() <= current_data_date)]
                            pct_change = recent_data_for_vol.pct_change().dropna()
                            volatility.append(pct_change.std() if not pct_change.empty else 0.02)

                         else:
                            volatility.append(0.02)
                    else:
                         volatility.append(0.02)

                except Exception as e:
                    logging.error(f"Error calculating volatility for {symbol}: {str(e)}")
                    volatility.append(0.02)
            else:
                volatility.append(0.02)
        return np.maximum(0.2, 1 - np.array(volatility, dtype=np.float64) * 5)


    def compute_sha_metric(self, timestamp_idx, amount=100000):
        """Compute the fragmentation obstruction value (Ш)"""
        try:
            liquidity_row = self.market_data['eth_liquidity'].iloc[timestamp_idx]
            liquidity_depths = np.array([liquidity_row[chain] for chain in self.chains], dtype=np.float64)

            log_liquidity = np.log(liquidity_depths + 1e-10)
            log_demand = np.log(np.array([amount] * len(self.chains), dtype=np.float64) + 1e-10)

            obstruction = np.linalg.norm(
                np.outer(log_liquidity, log_liquidity) - \
                np.outer(log_demand, log_demand)
            )

            return obstruction, liquidity_depths
        except Exception as e:
            logging.error(f"Error computing sha_metric: {str(e)}")
            default_liquidity = np.array([2.5e9, 1.8e9, 8e8, 1.2e9, 6e8], dtype=np.float64)
            return 1.0, default_liquidity

    def optimize_routing(self, timestamp_idx, amount=250000):
        """Enhanced routing with volatility adjustment"""
        try:
            _, liquidity_depths = self.compute_sha_metric(timestamp_idx, amount)
            volatility_factors = self.get_volatility_factors(timestamp_idx)

            if len(volatility_factors) != len(liquidity_depths):
                logging.warning(f"Volatility factor shape mismatch: {len(volatility_factors)} vs {len(liquidity_depths)}")
                volatility_factors = np.ones(len(liquidity_depths), dtype=np.float64) * 0.5

            if np.sum(liquidity_depths) == 0:
                return np.ones(len(self.chains), dtype=np.float64) / len(self.chains), liquidity_depths

            weights = liquidity_depths * (1 - volatility_factors * 0.5)
            weights = np.array(weights, dtype=np.float64)
            sum_weights = np.sum(weights)
            if sum_weights == 0:
                weights = np.ones(len(self.chains), dtype=np.float64) / len(self.chains)
            else:
                weights = weights / sum_weights

            weights[0] *= 1.5
            weights = weights * np.maximum(0.99, (1 + np.random.randn(len(weights)) * 0.01))
            weights = np.abs(weights)
            weights = weights / np.sum(weights)

            weights = np.clip(weights, 0.05, 0.5)
            weights = weights / np.sum(weights)

            return weights, liquidity_depths
        except Exception as e:
            logging.error(f"Error in optimize_routing: {str(e)}")
            return np.ones(len(self.chains), dtype=np.float64) / len(self.chains), np.array([2.5e9, 1.8e9, 8e8, 1.2e9, 6e8], dtype=np.float64)


    def calculate_slippage(self, weights, liquidity_depths, amount):
        """Calculate slippage for a given routing configuration"""
        executed = weights * amount
        slippage = 0
        for i, depth in enumerate(liquidity_depths):
            if depth > 0:
                price_impact = executed[i] / (depth + executed[i])
                slippage += price_impact * (1 + 0.5 * (amount / 1e6)) * (1e8 / (depth + 1e8))
            else:
                slippage += 1.0
        slippage = min(slippage * 100, 10.0)
        return slippage

# --- FastAPI Application Setup ---

app = FastAPI(
    title="Ш-Oracle Protocol API",
    description="API for Cohomological Liquidity Routing",
    version="1.0.0",
)

@app.on_event("startup")
async def startup_event():
    global market_data_global, oracle
    try:
        market_data_global = load_market_data_for_api()
        logging.info("\n✅ Market data loaded successfully into global variable.")
        oracle = ShaOracle(market_data_global)
    except Exception as e:
        logging.critical(f"\n✘ Failed to load market data: {str(e)}")
        raise RuntimeError("Failed to load essential market data for the API.") from e

@app.get("/", summary="Health Check", response_model=Dict[str, str])
async def root():
    """Basic health check endpoint."""
    return {"status": "ok", "message": "Ш-Oracle API is running"}

@app.post("/api/v1/optimize_routing", response_model=RoutingResponse, summary="Optimize Routing")
async def optimize_routing_endpoint(request: OptimizeRoutingRequest):
    """
    Find optimal cross-chain routing path for a given swap amount at a specific time index.
    """
    logging.info(f"Received optimize_routing request: timestamp_idx={request.timestamp_idx}, amount={request.amount}")
    try:
        if market_data_global is None or oracle is None:
            raise HTTPException(status_code=503, detail="API service is not ready: Market data not loaded.")
        if request.timestamp_idx < 0 or request.timestamp_idx >= len(oracle.market_data['eth_liquidity']):
            logging.error(f"Invalid timestamp_idx: {request.timestamp_idx}")
            raise HTTPException(status_code=400, detail="Invalid timestamp_idx: Must be within the bounds of loaded liquidity data.")
        if request.amount <= 0:
            logging.error(f"Invalid amount: {request.amount}")
            raise HTTPException(status_code=400, detail="Invalid amount: Must be a positive number.")

        optimal_weights, liquidity_depths = oracle.optimize_routing(
            request.timestamp_idx, request.amount
        )

        response_data = RoutingResponse(
            optimal_routing=optimal_weights.tolist(),
            liquidity_depths=liquidity_depths.tolist(),
            chains=oracle.chains,
            amount=request.amount
        )
        logging.info("Successfully processed optimize_routing request.")
        return response_data
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.exception(f"Internal Server Error during optimize_routing: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

@app.post("/api/v1/compute_sha", response_model=SHAResponse, summary="Compute Fragmentation Obstruction (Ш)")
async def compute_sha_endpoint(request: ComputeSHARequest):
    """
    Compute the fragmentation obstruction (Ш) metric and liquidity depths for a given swap amount at a specific time index.
    """
    logging.info(f"Received compute_sha request: timestamp_idx={request.timestamp_idx}, amount={request.amount}")
    try:
        if market_data_global is None or oracle is None:
            raise HTTPException(status_code=503, detail="API service is not ready: Market data not loaded.")
        if request.timestamp_idx < 0 or request.timestamp_idx >= len(oracle.market_data['eth_liquidity']):
            logging.error(f"Invalid timestamp_idx: {request.timestamp_idx}")
            raise HTTPException(status_code=400, detail="Invalid timestamp_idx: Must be within the bounds of loaded liquidity data.")
        if request.amount <= 0:
            logging.error(f"Invalid amount: {request.amount}")
            raise HTTPException(status_code=400, detail="Invalid amount: Must be a positive number.")

        sha_metric, liquidity_depths = oracle.compute_sha_metric(
            request.timestamp_idx, request.amount
        )

        response_data = SHAResponse(
            sha_metric=sha_metric,
            liquidity_depths=liquidity_depths.tolist(),
            chains=oracle.chains,
            amount=request.amount
        )
        logging.info("Successfully processed compute_sha request.")
        return response_data
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.exception(f"Internal Server Error during compute_sha: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

logging.info("FastAPI endpoints implemented and Oracle integrated.")
