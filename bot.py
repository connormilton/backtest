#!/usr/bin/env python3
"""
Self-Evolving LLM Forex Trading Bot with Backtesting Integration
Connects to the backtesting engine to validate strategies before live trading
"""

import os
import json
import time
import logging
import datetime
import pandas as pd
import numpy as np
import requests
import sys
from typing import Dict, List, Any, Optional, Tuple, Union
from dotenv import load_dotenv

# Import backtesting engine components
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest_engine import (
    DataProvider, Strategy, Backtester, BacktestResult, TradeResult,
    CSVDataProvider, PolygonDataProvider
)

# Import the LLM brain and memory components
from brain import LLMTradingBrain, TradingMemory, LLMGeneratedStrategy

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LLM_Trader")

# Create output directories
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("strategies", exist_ok=True)

# API credentials from environment - Use alternate variable names like in COINTOSS
OANDA_API_TOKEN = os.getenv("OANDA_API_KEY") or os.getenv("OANDA_API_TOKEN")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
OANDA_PRACTICE = os.getenv("OANDA_PRACTICE", "True").lower() in ["true", "1", "yes"]
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# Hard-coded parameters
MAX_DAILY_DRAWDOWN = 0.10  # 10% max daily drawdown
MAX_ACCOUNT_RISK = 0.10  # 10% max total account risk
INITIAL_SAFETY_LEVEL = 0.01  # Starting at 1% risk per trade
TARGET_DAILY_RETURN = 0.05  # 5% daily return target (reduced from 10%)
INCREASE_FACTOR = 0.0025  # Increase safety level by 0.25% after profitable trade
MIN_CONFIDENCE_THRESHOLD = 0.6  # Minimum backtesting confidence to proceed with trade

# --- Main Trading Bot ---
class EURUSDTradingBot:
    """Self-evolving EUR/USD trading bot powered by LLM and backtesting"""
    
    def __init__(self):
        """Initialize the trading bot"""
        try:
            # Initialize components
            self.oanda_client = OandaClient()
            self.oanda_data_provider = OandaDataProvider(OANDA_API_TOKEN, OANDA_ACCOUNT_ID, OANDA_PRACTICE)
            
            # Set up backtester with appropriate data provider
            if POLYGON_API_KEY:
                logger.info("Using Polygon.io for backtesting data")
                polygon_data_provider = PolygonDataProvider(POLYGON_API_KEY)
                self.backtester = Backtester(polygon_data_provider)
            else:
                logger.info("Using OANDA for backtesting data")
                self.backtester = Backtester(self.oanda_data_provider)
            
            self.memory = TradingMemory()
            self.brain = LLMTradingBrain(self.memory, self.backtester)
            
            # Initialize account info
            account = self.oanda_client.get_account()
            balance = float(account.get('balance', 1000))
            
            # Reset daily stats if needed
            self.memory.reset_daily_stats(balance)
            
            logger.info(f"EUR/USD Trading Bot initialized. Balance: {balance} {account.get('currency')}")
        except Exception as e:
            logger.error(f"Error initializing trading bot: {e}")
            raise
    
    def update_account_status(self):
        """Update account status and check daily limits"""
        account = self.oanda_client.get_account()
        balance = float(account.get('balance', 1000))
        
        # Update daily tracking
        if balance > self.memory.memory["daily_high_balance"]:
            self.memory.memory["daily_high_balance"] = balance
        
        # Calculate daily P&L
        self.memory.memory["daily_profit"] = balance - self.memory.memory["daily_starting_balance"]
        self.memory.memory["daily_profit_pct"] = (balance / self.memory.memory["daily_starting_balance"] - 1) * 100
        
        # Calculate drawdown
        if self.memory.memory["daily_high_balance"] > 0:
            current_drawdown = (self.memory.memory["daily_high_balance"] - balance) / self.memory.memory["daily_high_balance"] * 100
            self.memory.memory["daily_drawdown"] = max(self.memory.memory["daily_drawdown"], current_drawdown)
        
        # Save updated memory
        self.memory.save_memory()
        
        return {
            "balance": balance,
            "daily_profit_pct": self.memory.memory["daily_profit_pct"],
            "daily_drawdown": self.memory.memory["daily_drawdown"],
            "max_drawdown_reached": self.memory.memory["daily_drawdown"] >= MAX_DAILY_DRAWDOWN,
            "target_reached": self.memory.memory["daily_profit_pct"] >= TARGET_DAILY_RETURN
        }
    
    def execute_decision(self, decision, price_data):
        """Execute trading decision from LLM with enhanced risk management and backtesting validation"""
        action = decision.get("action", "WAIT")
        
        if action == "OPEN":
            # Extract trade details
            trade_details = decision.get("trade_details", {})
            direction = trade_details.get("direction")
            entry_price = trade_details.get("entry_price")
            stop_loss = trade_details.get("stop_loss")
            take_profit_levels = trade_details.get("take_profit", [])
            risk_percent = trade_details.get("risk_percent", 2.0)
            trailing_stop_distance = trade_details.get("trailing_stop_distance")
            
            # Get exit strategy for early management
            exit_strategy = decision.get("exit_strategy", {})
            early_exit_conditions = exit_strategy.get("early_exit_conditions")
            
            # Validate required fields
            if not all([direction, entry_price, stop_loss]):
                logger.warning(f"Missing required trade details: {trade_details}")
                return False
            
            # Check if backtest validation was already performed by EnhancedTradingBrain
            if "backtest_validation" in decision and decision["backtest_validation"].get("valid", False):
                # Trust the enhanced brain's validation
                validation_result = {"valid": True}
                logger.info("Using existing backtest validation result")
            else:
                # Fall back to traditional validation
                validation_result = self.brain.validate_trade(trade_details, price_data)
                logger.info(f"Trade validation result: {validation_result}")
            
            if not validation_result.get("valid", False):
                logger.warning(f"Trade validation failed: {validation_result.get('reason', 'Unknown reason')}")
                
                # Log the rejected trade for learning
                self.memory.log_trade({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "action": "REJECTED",
                    "direction": direction,
                    "entry_price": float(entry_price),
                    "stop_loss": float(stop_loss),
                    "take_profit_levels": take_profit_levels,
                    "trailing_stop_distance": trailing_stop_distance,
                    "risk_percent": float(risk_percent),
                    "strategy": trade_details.get("strategy", "unknown"),
                    "reasoning": trade_details.get("reasoning", ""),
                    "rejection_reason": validation_result.get("reason", "Failed validation"),
                    "is_win": None,
                    "is_loss": None
                })
                
                return False
            
            # Get account info for position sizing
            account = self.oanda_client.get_account()
            balance = float(account.get('balance', 1000))
            
            # Convert values to float
            try:
                entry_price = float(entry_price)
                stop_loss = float(stop_loss)
                
                # Handle risk percent which might be a string like "2.5" or "2"
                if isinstance(risk_percent, str):
                    # Remove any non-numeric characters except decimal point
                    risk_percent = ''.join(c for c in risk_percent if c.isdigit() or c == '.')
                    risk_percent = float(risk_percent) if risk_percent else 2.0
                elif isinstance(risk_percent, (int, float)):
                    risk_percent = float(risk_percent)
                else:
                    risk_percent = 2.0
                
                # Cap risk percent between 1-5%
                risk_percent = max(1.0, min(5.0, risk_percent))
                
                # Make sure take_profit is a list of floats
                if isinstance(take_profit_levels, list):
                    take_profit = [float(tp) for tp in take_profit_levels if tp]
                elif take_profit_levels and isinstance(take_profit_levels, (int, float, str)):
                    take_profit = [float(take_profit_levels)]
                else:
                    take_profit = []
                
                # Process trailing stop if provided
                if trailing_stop_distance:
                    if isinstance(trailing_stop_distance, str):
                        trailing_stop_distance = float(''.join(c for c in trailing_stop_distance if c.isdigit() or c == '.'))
                    elif isinstance(trailing_stop_distance, (int, float)):
                        trailing_stop_distance = float(trailing_stop_distance)
                    else:
                        trailing_stop_distance = None
                    
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting trade values to numeric: {e}")
                logger.error(f"Raw values - entry_price: {entry_price}, stop_loss: {stop_loss}, risk_percent: {risk_percent}")
                return False
            
            # Calculate position size based on risk
            risk_amount = balance * (risk_percent / 100)
            stop_distance = abs(entry_price - stop_loss)
            
            if stop_distance <= 0.0001:  # Minimum stop distance
                logger.error("Stop distance is too small, using minimum value")
                stop_distance = 0.0001
                
            # Basic position sizing (very simplified)
            units = int(risk_amount / stop_distance * 10000)  # Scaled for EUR/USD
            
            # Apply reasonable limits FIRST before checking margin
            units = min(units, 10000)  # Maximum 0.1 standard lot (reduced from 1.0)
            units = max(units, 1000)   # Minimum 0.01 lots (micro)
            
            # THEN check margin with the limited position size
            margin_available = self.oanda_client.get_margin_available()
            margin_needed = units * entry_price * 0.03  # Using 3% instead of 2% for safety
            
            if margin_needed > margin_available:
                logger.warning(f"Insufficient margin: needed {margin_needed:.2f}, available {margin_available:.2f}")
                # More conservative reduction - leave 30% buffer
                max_units = int((margin_available * 0.7) / (entry_price * 0.03))
                # Ensure it doesn't go below minimum
                max_units = max(max_units, 1000)
                units = min(units, max_units)
                logger.info(f"Reduced position size to {units} units due to margin constraints")
                
                # Double-check if we have enough margin with the reduced size
                new_margin_needed = units * entry_price * 0.03
                if new_margin_needed > margin_available:
                    logger.warning(f"Still insufficient margin after reduction. Skipping trade.")
                    return False
            
            logger.info(f"Calculated position size: {units} units based on risk {risk_percent}% and stop distance {stop_distance}")
            
            # Execute trade with all risk management parameters
            result = self.oanda_client.execute_trade(
                direction=direction,
                units=units,
                stop_loss=stop_loss,
                take_profit_levels=take_profit,
                trailing_stop_distance=trailing_stop_distance
            )
            
            # Process the result
            if "orderFillTransaction" in result:
                logger.info(f"Trade executed: {direction} EUR/USD, {units} units")
                fill = result["orderFillTransaction"]
                
                # Log trade details with enhanced information
                self.memory.log_trade({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "action": "OPEN",
                    "direction": direction,
                    "entry_price": float(fill.get("price", entry_price)),
                    "stop_loss": float(stop_loss),
                    "take_profit_levels": take_profit,
                    "trailing_stop_distance": trailing_stop_distance,
                    "units": units,
                    "risk_percent": float(risk_percent),
                    "strategy": trade_details.get("strategy", "unknown"),
                    "reasoning": trade_details.get("reasoning", ""),
                    "early_exit_conditions": early_exit_conditions,
                    "validation_confidence": validation_result.get("confidence", 0),
                    "is_win": None,  # Will be updated when closed
                    "is_loss": None  # Will be updated when closed
                })
                return True
            elif "orderCancelTransaction" in result:
                cancel = result["orderCancelTransaction"]
                reason = cancel.get("reason", "Unknown")
                logger.error(f"Order cancelled: {reason}")
                return False
            elif "error" in result:
                logger.error(f"Trade execution failed: {result['error']}")
                if "details" in result:
                    logger.error(f"Error details: {result['details']}")
                return False
            else:
                logger.error(f"Unknown trade execution result: {result}")
                return False
                
        elif action == "UPDATE":
            # Extract update details
            update_details = decision.get("update_details", {})
            new_stop_loss = update_details.get("new_stop_loss")
            
            if new_stop_loss:
                # Ensure numeric
                try:
                    new_stop_loss = float(new_stop_loss)
                except (ValueError, TypeError):
                    logger.error(f"Invalid stop loss value: {new_stop_loss}")
                    return False
                
                # Update stop loss
                result = self.oanda_client.update_stop_loss(new_stop_loss)
                
                if not isinstance(result, dict) or "error" not in result:
                    logger.info(f"Stop loss updated to {new_stop_loss}")
                    
                    # Log the update
                    self.memory.log_trade({
                        "timestamp": datetime.datetime.now().isoformat(),
                        "action": "UPDATE",
                        "new_stop_loss": float(new_stop_loss),
                        "reason": update_details.get("reason", "")
                    })
                    return True
                else:
                    logger.error(f"Stop loss update failed: {result}")
                    return False
                    
        elif action == "CREATE":
            # Extract strategy details
            strategy_details = decision.get("strategy_details", {})
            
            if not strategy_details:
                logger.warning("No strategy details provided for CREATE action")
                return False
            
            # Validate the new strategy with backtesting
            validation_result = self.brain.validate_strategy(
                strategy_details, 
                price_data, 
                risk_level=0.02, 
                validation_period=int(strategy_details.get("backtest_period", 30))
            )
            
            logger.info(f"Strategy validation result: {validation_result}")
            
            if validation_result.get("valid", False):
                logger.info(f"New strategy validated and saved: {strategy_details.get('name', 'Unnamed')}")
                return True
            else:
                logger.warning(f"Strategy validation failed: {validation_result.get('reason', 'Unknown reason')}")
                return False
        
        # Implement early exit logic if positions need to be closed based on conditions
        if "exit_strategy" in decision and action != "OPEN":
            positions = self.oanda_client.get_open_positions()
            eur_usd_positions = [p for p in positions if p.get("instrument") == "EUR_USD"]
            
            if eur_usd_positions and "early_exit_conditions" in decision.get("exit_strategy", {}):
                logger.info(f"Implementing early exit strategy")
                # Close positions
                for position in eur_usd_positions:
                    self.oanda_client.close_position(position.get("id"))
                    logger.info(f"Closed position based on exit strategy")
                return True
        
        return False

# --- Data Providers ---
class OandaDataProvider(DataProvider):
    """Data provider for OANDA API for backtesting recent data"""
    
    def __init__(self, api_token: str, account_id: str, practice: bool = True):
        """Initialize OANDA data provider"""
        self.api_token = api_token
        self.account_id = account_id
        # Update to use /v3 in the URL as seen in COINTOSS example
        self.base_url = "https://api-fxpractice.oanda.com/v3" if practice else "https://api-fxtrade.oanda.com/v3"
        
        # Set headers for all requests - Add space after Bearer as in COINTOSS
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        # Create session for requests
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Configure backup CSV provider
        self.csv_provider = CSVDataProvider("backtest_data")
    
    def _parse_timeframe(self, timeframe: str) -> str:
        """Convert generic timeframe to OANDA format"""
        # Map common format to OANDA format
        timeframe_map = {
            "minute": "M1",
            "5minute": "M5",
            "15minute": "M15",
            "hour": "H1",
            "4hour": "H4",
            "day": "D",
            "week": "W",
            "month": "M",
            # Already in OANDA format
            "M1": "M1",
            "M5": "M5",
            "M15": "M15",
            "M30": "M30",
            "H1": "H1",
            "H2": "H2",
            "H4": "H4",
            "H6": "H6",
            "H8": "H8",
            "H12": "H12",
            "D": "D",
            "W": "W",
            "M": "M"
        }
        
        return timeframe_map.get(timeframe, "H1")  # Default to H1 if unknown
    
    def get_historical_data(self, instrument: str, start_date: Union[str, datetime.datetime], 
                           end_date: Union[str, datetime.datetime], timeframe: str = "day") -> pd.DataFrame:
        """Fetch historical OHLCV data for the specified instrument and timeframe"""
        # Convert instrument format if needed (EUR/USD -> EUR_USD)
        oanda_instrument = instrument.replace("/", "_")
        
        # Convert timeframe to OANDA format
        oanda_timeframe = self._parse_timeframe(timeframe)
        
        # Calculate how many candles needed based on dates
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Calculate time difference
        time_diff = end_date - start_date
        
        # Calculate count based on timeframe and time difference
        if oanda_timeframe.startswith("M"):
            # Minutes
            minutes = int(oanda_timeframe[1:]) if len(oanda_timeframe) > 1 else 1
            count = int(time_diff.total_seconds() / 60 / minutes) + 10  # Add buffer
        elif oanda_timeframe.startswith("H"):
            # Hours
            hours = int(oanda_timeframe[1:]) if len(oanda_timeframe) > 1 else 1
            count = int(time_diff.total_seconds() / 3600 / hours) + 5  # Add buffer
        elif oanda_timeframe == "D":
            # Days
            count = time_diff.days + 5  # Add buffer
        elif oanda_timeframe == "W":
            # Weeks
            count = int(time_diff.days / 7) + 2  # Add buffer
        elif oanda_timeframe == "M":
            # Months
            count = int(time_diff.days / 30) + 2  # Add buffer
        else:
            # Default to days
            count = time_diff.days + 5
        
        # OANDA API limits count to 5000
        count = min(count, 5000)
        
        # First try to get data from OANDA API
        try:
            # Build API endpoint - Path already has /v3
            endpoint = f"{self.base_url}/instruments/{oanda_instrument}/candles"
            
            params = {
                "count": count,
                "granularity": oanda_timeframe,
                "price": "M",  # Midpoint candlesticks
                "from": start_date.isoformat() if hasattr(start_date, 'isoformat') else start_date,
                "to": end_date.isoformat() if hasattr(end_date, 'isoformat') else end_date
            }
            
            # Make API request
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check if candles exist
            if "candles" not in data or not data["candles"]:
                logger.warning(f"No candles returned for {instrument} from OANDA API")
                return self.csv_provider.get_historical_data(instrument, start_date, end_date, timeframe)
            
            # Process candles into DataFrame
            records = []
            for candle in data["candles"]:
                # Skip incomplete candles
                if not candle.get("complete", False):
                    continue
                
                mid = candle.get("mid", {})
                records.append({
                    "datetime": pd.to_datetime(candle.get("time")),
                    "open": float(mid.get("o", 0)),
                    "high": float(mid.get("h", 0)),
                    "low": float(mid.get("l", 0)),
                    "close": float(mid.get("c", 0)),
                    "volume": int(candle.get("volume", 0))
                })
            
            # Create DataFrame
            df = pd.DataFrame(records)
            
            if not df.empty:
                df.set_index("datetime", inplace=True)
                
                # Filter to exact date range
                df = df[(df.index >= start_date) & (df.index <= end_date)]
                
                # Save to CSV for future use
                csv_path = f"backtest_data/{instrument.replace('/', '_')}_{timeframe}.csv"
                df.to_csv(csv_path)
                
                return df
            else:
                logger.warning(f"Empty DataFrame after processing OANDA data for {instrument}")
                return self.csv_provider.get_historical_data(instrument, start_date, end_date, timeframe)
                
        except Exception as e:
            logger.error(f"Error fetching data from OANDA: {str(e)}")
            # Fallback to CSV provider
            return self.csv_provider.get_historical_data(instrument, start_date, end_date, timeframe)
    
    def get_instruments(self) -> List[str]:
        """Get list of available instruments"""
        try:
            # Update path to include /v3
            endpoint = f"{self.base_url}/accounts/{self.account_id}/instruments"
            
            response = self.session.get(endpoint)
            response.raise_for_status()
            
            data = response.json()
            
            if "instruments" not in data:
                logger.warning("No instruments found in OANDA API response")
                return []
            
            instruments = []
            for instrument in data["instruments"]:
                name = instrument.get("name", "")
                if name:
                    # Convert OANDA format to standard format (EUR_USD -> EUR/USD)
                    standard_name = name.replace("_", "/")
                    instruments.append(standard_name)
            
            return instruments
            
        except Exception as e:
            logger.error(f"Error fetching instruments from OANDA: {str(e)}")
            return []
    
    def get_latest_data(self, instrument: str, count: int = 1, timeframe: str = "day") -> pd.DataFrame:
        """Get the latest N data points for an instrument"""
        # Convert instrument format if needed (EUR/USD -> EUR_USD)
        oanda_instrument = instrument.replace("/", "_")
        
        # Convert timeframe to OANDA format
        oanda_timeframe = self._parse_timeframe(timeframe)
        
        try:
            # Build API endpoint - Path already has /v3
            endpoint = f"{self.base_url}/instruments/{oanda_instrument}/candles"
            
            params = {
                "count": count,
                "granularity": oanda_timeframe,
                "price": "M"  # Midpoint candlesticks
            }
            
            # Make API request
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check if candles exist
            if "candles" not in data or not data["candles"]:
                logger.warning(f"No candles returned for {instrument} from OANDA API")
                return pd.DataFrame()
            
            # Process candles into DataFrame
            records = []
            for candle in data["candles"]:
                mid = candle.get("mid", {})
                records.append({
                    "datetime": pd.to_datetime(candle.get("time")),
                    "open": float(mid.get("o", 0)),
                    "high": float(mid.get("h", 0)),
                    "low": float(mid.get("l", 0)),
                    "close": float(mid.get("c", 0)),
                    "volume": int(candle.get("volume", 0))
                })
            
            # Create DataFrame
            df = pd.DataFrame(records)
            
            if not df.empty:
                df.set_index("datetime", inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching latest data from OANDA: {str(e)}")
            return pd.DataFrame()

# --- OANDA API Client ---
class OandaClient:
    """OANDA API client for forex trading execution"""
    
    def __init__(self):
        """Initialize OANDA API client with improved connection handling"""
        # Create session with connection pooling
        self.base_url = "https://api-fxpractice.oanda.com/v3" if OANDA_PRACTICE else "https://api-fxtrade.oanda.com/v3"
        self.headers = {
            "Authorization": f"Bearer {OANDA_API_TOKEN}",
            "Content-Type": "application/json"
        }
        # Initialize cache containers
        self._data_cache = {}
        self._last_account_info = {}
        # Create robust session
        self.session = self._create_session()
        # Verify connection
        self._test_connection()
    
    def _create_session(self):
        """Create a robust session with connection pooling"""
        session = requests.Session()
        session.headers.update(self.headers)
        # Configure connection pooling and retries
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=5,
            pool_maxsize=10,
            max_retries=3,
            pool_block=False
        )
        session.mount('https://', adapter)
        return session
    
    def _test_connection(self):
        """Test connection to OANDA API"""
        account = self.get_account()
        logger.info(f"Connected to OANDA. Balance: {account.get('balance')} {account.get('currency')}")
    
    def get_account(self):
        """Get account information with retry logic and caching"""
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(f"{self.base_url}/accounts/{OANDA_ACCOUNT_ID}/summary")
                response.raise_for_status()
                account_info = response.json().get("account", {})
                # Cache the successful result
                self._last_account_info = account_info
                return account_info
            except Exception as e:
                logger.warning(f"Error getting account (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    # Reset session to handle connection issues
                    self.session = self._create_session()
                else:
                    logger.error(f"Failed to get account after {max_retries} attempts")
        
        # Return cached account info if available, otherwise empty dict with sane defaults
        if not self._last_account_info:
            self._last_account_info = {"balance": 1000, "currency": "GBP"}
        return self._last_account_info
    
    def get_margin_available(self):
        """Get available margin from the account"""
        try:
            account = self.get_account()
            margin_available = float(account.get('marginAvailable', 0))
            return margin_available
        except Exception as e:
            logger.error(f"Error getting margin available: {e}")
            return 0
    
    def get_eur_usd_data(self, count=100, granularity="H1"):
        """Get EUR/USD price data with caching"""
        cache_key = f"EUR_USD_{granularity}"
        current_time = datetime.datetime.now()
        
        # Determine cache validity period based on timeframe
        if granularity.startswith('M'):
            cache_validity_minutes = 1  # 1 minute for minute-level data
        elif granularity.startswith('H'):
            cache_validity_minutes = 10  # 10 minutes for hourly data
        else:
            cache_validity_minutes = 60  # 1 hour for daily+ data
        
        # Check if we have cached data that's recent enough
        if hasattr(self, '_data_cache') and cache_key in self._data_cache:
            cached_data, timestamp = self._data_cache[cache_key]
            # Only use cached data if it's recent enough
            if (current_time - timestamp).total_seconds() < cache_validity_minutes * 60:
                return cached_data
        
        # If no valid cache, request from API
        try:
            params = {
                "count": count,
                "granularity": granularity,
                "price": "M"
            }
            response = self.session.get(
                f"{self.base_url}/instruments/EUR_USD/candles", 
                params=params
            )
            response.raise_for_status()
            candles = response.json().get("candles", [])
            
            # Convert to pandas DataFrame
            data = []
            for candle in candles:
                if candle.get("complete", False):
                    mid = candle.get("mid", {})
                    data.append({
                        "time": candle.get("time"),
                        "open": float(mid.get("o", 0)),
                        "high": float(mid.get("h", 0)),
                        "low": float(mid.get("l", 0)),
                        "close": float(mid.get("c", 0)),
                        "volume": int(candle.get("volume", 0)),
                        "timeframe": granularity
                    })
            df = pd.DataFrame(data)
            
            # Cache the result
            if not hasattr(self, '_data_cache'):
                self._data_cache = {}
            self._data_cache[cache_key] = (df, current_time)
            
            return df
        except Exception as e:
            # If we have any cached data, return it rather than failing
            if hasattr(self, '_data_cache') and cache_key in self._data_cache:
                logger.warning(f"Using cached data for {cache_key} due to API error: {e}")
                return self._data_cache[cache_key][0]
            logger.error(f"Error getting EUR/USD data: {e}")
            return pd.DataFrame()
            
    def get_multi_timeframe_data(self):
        """Get EUR/USD data across multiple timeframes with reduced loading"""
        # Define the base timeframes we need every time
        primary_timeframes = {
            "H1": {"count": 100, "name": "1-hour"}
        }
        
        # Only add additional timeframes every other cycle
        if not hasattr(self, '_timeframe_cycle') or self._timeframe_cycle % 2 == 0:
            secondary_timeframes = {
                "M15": {"count": 100, "name": "15-minute"},
                "H4": {"count": 50, "name": "4-hour"}
            }
            primary_timeframes.update(secondary_timeframes)
        
        # And these timeframes even less frequently
        if not hasattr(self, '_timeframe_cycle') or self._timeframe_cycle % 4 == 0:
            tertiary_timeframes = {
                "M5": {"count": 100, "name": "5-minute"},
                "D": {"count": 30, "name": "Daily"}
            }
            primary_timeframes.update(tertiary_timeframes)
        
        # Toggle cycle
        self._timeframe_cycle = getattr(self, '_timeframe_cycle', 0) + 1
        if self._timeframe_cycle > 10:
            self._timeframe_cycle = 0
        
        # Fetch data
        data = {}
        for tf, tf_info in primary_timeframes.items():
            df = self.get_eur_usd_data(count=tf_info["count"], granularity=tf)
            if not df.empty:
                # Only include the most recent data points
                data[tf_info["name"]] = df.tail(10).to_dict('records')
        
        return data
    
    def get_economic_calendar(self):
        """Get recent and upcoming economic events that might impact EUR/USD"""
        try:
            # In a real implementation, this would call an economic calendar API
            # For now, just fetch from ForexFactory or similar sites
            # We'll simulate some data
            import datetime
            
            today = datetime.datetime.now()
            
            # Simulate some economic events
            return {
                "recent_events": [
                    {
                        "date": (today - datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
                        "time": "10:00",
                        "currency": "EUR",
                        "impact": "High",
                        "event": "ECB Interest Rate Decision",
                        "actual": "4.00%",
                        "forecast": "4.00%",
                        "previous": "4.00%"
                    },
                    {
                        "date": (today - datetime.timedelta(days=2)).strftime("%Y-%m-%d"),
                        "time": "14:30",
                        "currency": "USD",
                        "impact": "High",
                        "event": "Nonfarm Payrolls",
                        "actual": "236K",
                        "forecast": "240K",
                        "previous": "315K"
                    }
                ],
                "upcoming_events": [
                    {
                        "date": (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
                        "time": "14:30",
                        "currency": "USD",
                        "impact": "High",
                        "event": "CPI m/m",
                        "forecast": "0.3%",
                        "previous": "0.4%"
                    },
                    {
                        "date": (today + datetime.timedelta(days=2)).strftime("%Y-%m-%d"),
                        "time": "10:00",
                        "currency": "EUR",
                        "impact": "Medium",
                        "event": "Industrial Production m/m",
                        "forecast": "0.5%",
                        "previous": "0.7%"
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Error getting economic calendar: {e}")
            return {"recent_events": [], "upcoming_events": []}
            
    def get_intermarket_data(self):
        """Get intermarket correlation data that affects EUR/USD"""
        try:
            # Simulate related market data
            import random
            
            # Sample data structure
            related_markets = {
                "currency_pairs": {
                    "GBP/USD": random.uniform(1.25, 1.30),
                    "USD/JPY": random.uniform(110.0, 115.0),
                    "USD/CHF": random.uniform(0.90, 0.95),
                    "AUD/USD": random.uniform(0.65, 0.70)
                },
                "commodities": {
                    "Gold": random.uniform(1800, 2000),
                    "Oil_WTI": random.uniform(70, 85)
                },
                "indices": {
                    "S&P500": random.uniform(4500, 4800),
                    "DAX": random.uniform(15000, 16000),
                    "FTSE": random.uniform(7500, 8000)
                },
                "bonds": {
                    "US_10Y_Yield": random.uniform(3.5, 4.2),
                    "DE_10Y_Yield": random.uniform(2.0, 2.5),
                    "US_DE_Spread": random.uniform(1.2, 1.8)
                },
                "correlations": {
                    "EURUSD_GBPUSD": random.uniform(0.7, 0.9),
                    "EURUSD_Gold": random.uniform(0.3, 0.6),
                    "EURUSD_US_DE_Spread": random.uniform(-0.7, -0.5)
                }
            }
            
            return related_markets
        except Exception as e:
            logger.error(f"Error getting intermarket data: {e}")
            return {}
            
    def get_technical_indicators(self, price_data):
        """Calculate common technical indicators for the price data"""
        try:
            if price_data.empty:
                return {}
                
            # Clone the dataframe to avoid modifying the original
            df = price_data.copy()
            
            # Calculate moving averages
            df['MA_20'] = df['close'].rolling(window=20).mean()
            df['MA_50'] = df['close'].rolling(window=50).mean()
            df['MA_100'] = df['close'].rolling(window=100).mean()
            
            # Calculate RSI (14-period)
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Calculate Bollinger Bands (20-period, 2 standard deviations)
            df['BB_Middle'] = df['close'].rolling(window=20).mean()
            df['BB_StdDev'] = df['close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_StdDev']
            df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_StdDev']
            
            # Calculate MACD
            df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # Calculate ATR (14-period)
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['ATR'] = true_range.rolling(14).mean()
            
            # Get the latest values for all indicators
            try:
                latest = df.iloc[-1].to_dict()
            except IndexError:
                logger.warning("Not enough data to calculate indicators, using empty values")
                return {
                    "moving_averages": {},
                    "oscillators": {},
                    "volatility": {},
                    "trend_strength": {},
                    "support_resistance": {"key_resistance_levels": [], "key_support_levels": []}
                }
            
            # Create a simplified structure with just the latest indicator values
            indicators = {
                "moving_averages": {
                    "MA_20": latest.get('MA_20'),
                    "MA_50": latest.get('MA_50'),
                    "MA_100": latest.get('MA_100'),
                },
                "oscillators": {
                    "RSI": latest.get('RSI'),
                    "MACD": latest.get('MACD'),
                    "MACD_Signal": latest.get('MACD_Signal'),
                    "MACD_Histogram": latest.get('MACD_Histogram')
                },
                "volatility": {
                    "ATR": latest.get('ATR'),
                    "BB_Width": latest.get('BB_Upper') - latest.get('BB_Lower') if pd.notna(latest.get('BB_Upper')) and pd.notna(latest.get('BB_Lower')) else None,
                    "BB_Upper": latest.get('BB_Upper'),
                    "BB_Lower": latest.get('BB_Lower')
                },
                "trend_strength": {
                    "Price_vs_MA20": (latest.get('close') / latest.get('MA_20') - 1) * 100 if pd.notna(latest.get('MA_20')) and latest.get('MA_20') > 0 else None,
                    "MA20_vs_MA50": (latest.get('MA_20') / latest.get('MA_50') - 1) * 100 if pd.notna(latest.get('MA_50')) and latest.get('MA_50') > 0 else None,
                },
                "support_resistance": {
                    "key_resistance_levels": [],
                    "key_support_levels": []
                }
            }
            
            # Add support/resistance levels
            try:
                highs = df['high'].nlargest(5).tolist()
                lows = df['low'].nsmallest(5).tolist()
                
                indicators["support_resistance"] = {
                    "key_resistance_levels": highs,
                    "key_support_levels": lows
                }
            except Exception as e:
                logger.warning(f"Error calculating support/resistance levels: {e}")
            
            return indicators
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {}
            
    def get_open_positions(self):
        """Get open positions"""
        try:
            # Update path to include /v3
            response = self.session.get(f"{self.base_url}/accounts/{OANDA_ACCOUNT_ID}/openPositions")
            response.raise_for_status()
            return response.json().get("positions", [])
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def execute_trade(self, direction, units, stop_loss=None, take_profit_levels=None, trailing_stop_distance=None):
        """Execute a trade on EUR/USD with simplified risk management using OANDA's native functionality"""
        try:
            # Determine units based on direction
            if direction.upper() == "SELL":
                units = -abs(units)
            else:  # BUY
                units = abs(units)
                
            # Build order request
            order_data = {
                "order": {
                    "instrument": "EUR_USD",
                    "units": str(units),
                    "timeInForce": "FOK",
                    "positionFill": "DEFAULT",
                    "type": "MARKET"
                }
            }
            
            # Add stop loss if provided
            if stop_loss is not None:
                order_data["order"]["stopLossOnFill"] = {
                    "price": str(stop_loss),
                    "timeInForce": "GTC"
                }
                
                # Add trailing stop if provided
                if trailing_stop_distance is not None:
                    order_data["order"]["trailingStopLossOnFill"] = {
                        "distance": str(trailing_stop_distance),
                        "timeInForce": "GTC"
                    }
            
            # Add take profit if provided (just the first level)
            if take_profit_levels and len(take_profit_levels) > 0:
                order_data["order"]["takeProfitOnFill"] = {
                    "price": str(take_profit_levels[0]),
                    "timeInForce": "GTC"
                }
            
            # Execute the order
            logger.info(f"Executing {direction} order for EUR_USD with {units} units")
            
            # Update path for /v3
            response = self.session.post(
                f"{self.base_url}/accounts/{OANDA_ACCOUNT_ID}/orders",
                json=order_data
            )
            
            # Handle response
            response.raise_for_status()
            result = response.json()
            
            # Check if the order was filled
            if "orderFillTransaction" in result:
                fill_txn = result["orderFillTransaction"]
                fill_id = fill_txn.get("id", "Unknown ID")
                trade_id = fill_txn.get("tradeOpened", {}).get("tradeID")
                logger.info(f"Order executed and filled: {fill_id}, Trade ID: {trade_id}")
                
            else:
                logger.info(f"Order created: {result.get('orderCreateTransaction', {}).get('id', 'Unknown ID')}")
                
            return result
            
        except requests.exceptions.HTTPError as e:
            # Handle HTTP errors from the API
            error_response = {}
            try:
                error_response = e.response.json()
                logger.error(f"OANDA API error: {error_response}")
            except:
                logger.error(f"OANDA API HTTP error: {str(e)}")
            
            return {"error": str(e), "details": error_response}
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {"error": str(e)}
            
    def update_stop_loss(self, stop_loss):
        """Update stop loss for EUR/USD position"""
        try:
            # Get open trades for EUR_USD
            # Update path for /v3
            response = self.session.get(
                f"{self.base_url}/accounts/{OANDA_ACCOUNT_ID}/trades?instrument=EUR_USD&state=OPEN"
            )
            response.raise_for_status()
            trades = response.json().get("trades", [])
            
            if not trades:
                logger.warning("No open EUR/USD trades found to update stop loss")
                return {"error": "No open trades found"}
            
            results = []
            for trade in trades:
                trade_id = trade["id"]
                update_data = {
                    "stopLoss": {
                        "price": str(stop_loss),
                        "timeInForce": "GTC"
                    }
                }
                
                try:
                    # Update path for /v3
                    update_response = self.session.put(
                        f"{self.base_url}/accounts/{OANDA_ACCOUNT_ID}/trades/{trade_id}/orders",
                        json=update_data
                    )
                    update_response.raise_for_status()
                    results.append(update_response.json())
                    logger.info(f"Updated stop loss for trade {trade_id} to {stop_loss}")
                except Exception as update_error:
                    logger.error(f"Error updating stop loss for trade {trade_id}: {update_error}")
                    results.append({"error": str(update_error), "trade_id": trade_id})
            
            return results
        except Exception as e:
            logger.error(f"Error updating stop loss: {e}")
            return {"error": str(e)}
    
    def close_position(self, position_id):
        """Close a specific position"""
        try:
            # Update path for /v3
            response = self.session.put(
                f"{self.base_url}/accounts/{OANDA_ACCOUNT_ID}/positions/EUR_USD/close",
                json={"longUnits": "ALL"}  # This will close all long positions
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Position closed: {result}")
            return result
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {"error": str(e)}
            
    def get_market_sentiment(self):
        """Get market sentiment data for EUR/USD
        This would normally come from an external API, but we'll simulate it"""
        try:
            import random
            
            # Simulate sentiment data
            bullish_percent = random.randint(30, 70)
            bearish_percent = 100 - bullish_percent
            
            # Volume indicators
            volume_status = random.choice(["High", "Average", "Low"])
            
            # Positioning data
            positioning = {
                "retail_long_percent": random.randint(30, 70),
                "retail_short_percent": random.randint(30, 70),
                "institutional_bias": random.choice(["Bullish", "Bearish", "Neutral"])
            }
            
            return {
                "pair": "EUR/USD",
                "timestamp": datetime.datetime.now().isoformat(),
                "sentiment": {
                    "bullish_percent": bullish_percent,
                    "bearish_percent": bearish_percent,
                    "overall": "Bullish" if bullish_percent > 55 else "Bearish" if bullish_percent < 45 else "Neutral"
                },
                "volume": {
                    "status": volume_status,
                    "relative_to_average": random.uniform(0.8, 1.2)
                },
                "positioning": positioning
            }
        except Exception as e:
            logger.error(f"Error getting market sentiment: {e}")
            return {}

# Main entry point
if __name__ == "__main__":
    try:
        # Create and run the trading bot
        trading_bot = EURUSDTradingBot()
        logger.info("Trading bot initialized successfully")
        
        # Main loop would go here...
        
    except Exception as e:
        logger.error(f"Error running trading bot: {e}")