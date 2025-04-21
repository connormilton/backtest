#!/usr/bin/env python3
"""
EUR/USD Trading Bot that uses LLM-based market analysis
"""

import os
import json
import time
import logging
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LLM_Trader")

# Load environment variables
load_dotenv()

# API Keys
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# Import after environment variables are loaded
from brain import LLMTradingBrain  # Fixed import name
from memory import TradingMemory
from backtest_engine import Backtester, CSVDataProvider, PolygonDataProvider

class OandaClient:
    """Client for interacting with OANDA's API"""
    
    def __init__(self, api_key, account_id):
        """Initialize OANDA client"""
        self.api_key = api_key
        self.account_id = account_id
        self.base_url = "https://api-fxpractice.oanda.com"  # Using practice API
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        logger.info("OANDA client initialized")
        
    def get_account_summary(self):
        """Get account summary from OANDA"""
        import requests
        
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/summary"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            account = data.get("account", {})
            
            # Extract relevant information
            summary = {
                "balance": float(account.get("balance", 0)),
                "currency": account.get("currency", "USD"),
                "unrealized_pl": float(account.get("unrealizedPL", 0)),
                "nav": float(account.get("NAV", 0)),
                "margin_rate": float(account.get("marginRate", 0)),
                "margin_used": float(account.get("marginUsed", 0)),
                "margin_available": float(account.get("marginAvailable", 0)),
                "open_trade_count": account.get("openTradeCount", 0),
                "pending_order_count": account.get("pendingOrderCount", 0)
            }
            
            logger.info(f"Connected to OANDA. Balance: {summary['balance']} {summary['currency']}")
            return summary
            
        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            return {
                "balance": 0,
                "currency": "USD",
                "unrealized_pl": 0,
                "error": str(e)
            }
    
    def get_eur_usd_data(self, count=100, granularity="H1"):
        """Get EUR/USD price data from OANDA"""
        import requests
        
        try:
            url = f"{self.base_url}/v3/instruments/EUR_USD/candles"
            params = {
                "count": count,
                "granularity": granularity,
                "price": "M"  # Midpoint
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            candles = data.get("candles", [])
            
            # Convert to DataFrame
            records = []
            for candle in candles:
                mid = candle.get("mid", {})
                records.append({
                    "datetime": pd.to_datetime(candle.get("time")),
                    "open": float(mid.get("o", 0)),
                    "high": float(mid.get("h", 0)),
                    "low": float(mid.get("l", 0)),
                    "close": float(mid.get("c", 0)),
                    "volume": int(candle.get("volume", 0))
                })
            
            df = pd.DataFrame(records)
            if not df.empty:
                df.set_index("datetime", inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting EUR/USD data: {e}")
            return pd.DataFrame()
    
    def get_multi_timeframe_data(self):
        """Get data across multiple timeframes for comprehensive analysis"""
        try:
            # Get various timeframes
            h4_data = self.get_eur_usd_data(count=50, granularity="H4")
            d_data = self.get_eur_usd_data(count=20, granularity="D")
            
            return {
                "h4": h4_data.to_dict() if not h4_data.empty else {"error": "No H4 data"},
                "daily": d_data.to_dict() if not d_data.empty else {"error": "No daily data"}
            }
            
        except Exception as e:
            logger.error(f"Error getting multi-timeframe data: {e}")
            return {"error": str(e)}
    
    def get_open_positions(self):
        """Get open positions from OANDA"""
        import requests
        
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/openPositions"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            positions = data.get("positions", [])
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting open positions: {e}")
            return []
    
    def create_order(self, instrument, units, price=None, stop_loss=None, take_profit=None):
        """Create a market or limit order with OANDA"""
        import requests
        
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/orders"
            
            # Determine order type
            order_type = "MARKET" if price is None else "LIMIT"
            
            # Prepare the order body
            order_body = {
                "order": {
                    "type": order_type,
                    "instrument": instrument,
                    "units": str(units),
                    "timeInForce": "GTC"  # Good 'til cancelled
                }
            }
            
            # Add price for limit orders
            if price is not None:
                order_body["order"]["price"] = str(price)
            
            # Add stop loss if provided
            if stop_loss is not None:
                order_body["order"]["stopLossOnFill"] = {
                    "price": str(stop_loss),
                    "timeInForce": "GTC"
                }
            
            # Add take profit if provided
            if take_profit is not None:
                order_body["order"]["takeProfitOnFill"] = {
                    "price": str(take_profit),
                    "timeInForce": "GTC"
                }
            
            # Send order request
            response = requests.post(url, headers=self.headers, json=order_body)
            response.raise_for_status()
            
            # Process response
            result = response.json()
            order_created = result.get("orderCreateTransaction", {})
            order_id = order_created.get("id")
            
            logger.info(f"Order created: {order_id} - {instrument} {units} units")
            return {
                "success": True,
                "order_id": order_id,
                "units": units,
                "price": price,
                "stop_loss": stop_loss,
                "take_profit": take_profit
            }
            
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def update_order(self, order_id, new_stop_loss=None, new_take_profit=None):
        """Update an existing order in OANDA"""
        import requests
        
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/orders/{order_id}"
            
            # Prepare the update body
            update_body = {
                "order": {
                    "id": order_id,
                    "timeInForce": "GTC"
                }
            }
            
            # Add new stop loss if provided
            if new_stop_loss is not None:
                update_body["order"]["stopLossOnFill"] = {
                    "price": str(new_stop_loss),
                    "timeInForce": "GTC"
                }
            
            # Add new take profit if provided
            if new_take_profit is not None:
                update_body["order"]["takeProfitOnFill"] = {
                    "price": str(new_take_profit),
                    "timeInForce": "GTC"
                }
            
            # Send update request
            response = requests.put(url, headers=self.headers, json=update_body)
            response.raise_for_status()
            
            # Process response
            result = response.json()
            
            logger.info(f"Order updated: {order_id}")
            return {
                "success": True,
                "order_id": order_id
            }
            
        except Exception as e:
            logger.error(f"Error updating order: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def close_position(self, instrument="EUR_USD"):
        """Close a position for a specific instrument"""
        import requests
        
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/positions/{instrument}/close"
            
            # Get the current position to determine units
            positions = self.get_open_positions()
            position = next((p for p in positions if p.get("instrument") == instrument), None)
            
            if not position:
                logger.warning(f"No open position found for {instrument}")
                return {
                    "success": False,
                    "error": f"No open position for {instrument}"
                }
            
            # Determine which side to close (long or short)
            long_units = int(position.get("long", {}).get("units", 0))
            short_units = int(position.get("short", {}).get("units", 0))
            
            # Prepare close request
            close_body = {}
            if long_units > 0:
                close_body["longUnits"] = "ALL"
            elif short_units > 0:
                close_body["shortUnits"] = "ALL"
            else:
                logger.warning(f"Position has 0 units for {instrument}")
                return {
                    "success": False,
                    "error": f"Position has 0 units for {instrument}"
                }
            
            # Send close request
            response = requests.put(url, headers=self.headers, json=close_body)
            response.raise_for_status()
            
            # Process response
            result = response.json()
            
            logger.info(f"Position closed for {instrument}")
            return {
                "success": True,
                "instrument": instrument
            }
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def update_position_stops(self, instrument, new_stop_loss=None, new_take_profit=None):
        """Update stop loss and take profit for an open position"""
        import requests
        
        try:
            # Get open positions to validate
            positions = self.get_open_positions()
            position = next((p for p in positions if p.get("instrument") == instrument), None)
            
            if not position:
                logger.warning(f"No open position found for {instrument}")
                return {
                    "success": False,
                    "error": f"No open position for {instrument}"
                }
            
            # Get the position ID
            position_id = position.get("id")
            
            # Build the URL
            url = f"{self.base_url}/v3/accounts/{self.account_id}/positions/{instrument}"
            
            # Prepare update body
            update_body = {}
            
            # Add stop loss if provided
            if new_stop_loss is not None:
                update_body["stopLoss"] = {
                    "price": str(new_stop_loss),
                    "timeInForce": "GTC"
                }
            
            # Add take profit if provided
            if new_take_profit is not None:
                update_body["takeProfit"] = {
                    "price": str(new_take_profit),
                    "timeInForce": "GTC"
                }
            
            # If nothing to update, return
            if not update_body:
                logger.warning(f"No updates provided for {instrument}")
                return {
                    "success": False,
                    "error": "No updates provided"
                }
            
            # Send update request
            response = requests.put(url, headers=self.headers, json=update_body)
            response.raise_for_status()
            
            # Process response
            result = response.json()
            
            logger.info(f"Position updated for {instrument}")
            return {
                "success": True,
                "instrument": instrument
            }
            
        except Exception as e:
            logger.error(f"Error updating position: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def calculate_position_size(self, account_balance, risk_percent, entry_price, stop_loss):
        """Calculate position size based on risk parameters"""
        # Convert risk percent to decimal
        risk_decimal = risk_percent / 100
        
        # Calculate risk amount in account currency
        risk_amount = account_balance * risk_decimal
        
        # Calculate price difference between entry and stop
        price_diff = abs(entry_price - stop_loss)
        
        # Calculate position size in units
        if price_diff > 0:
            position_size = risk_amount / price_diff
            # Round down to nearest whole number
            position_size = int(position_size)
        else:
            position_size = 0
            
        return position_size

class EURUSDTradingBot:
    """LLM-Powered EUR/USD Trading Bot"""
    
    def __init__(self):
        """Initialize the trading bot"""
        # Initialize OANDA client
        self.oanda_client = OandaClient(OANDA_API_KEY, OANDA_ACCOUNT_ID)
        
        # Get account info
        self.account_info = self.oanda_client.get_account_summary()
        
        # Initialize memory
        self.memory = TradingMemory()
        
        # Initialize backtester
        if POLYGON_API_KEY:
            self.data_provider = PolygonDataProvider(POLYGON_API_KEY)
            logger.info("Using Polygon.io for backtesting data")
        else:
            self.data_provider = CSVDataProvider()
            logger.info("Using CSV data provider for backtesting")
            
        self.backtester = Backtester(self.data_provider)
        
        # Initialize brain - use the name that matches your implementation
        self.brain = LLMTradingBrain(self.memory)
        
        # Configure risk management
        self.daily_target_pct = 2.0  # 2% daily profit target
        self.max_daily_drawdown_pct = 1.0  # 1% maximum daily drawdown
        self.max_risk_per_trade_pct = 1.0  # 1% maximum risk per trade
        
        # Track last decision time
        self.last_decision_time = None
        
        # Track errors
        self.last_error = None
        
        logger.info(f"EUR/USD Trading Bot initialized. Balance: {self.account_info['balance']} {self.account_info['currency']}")
    
    def update_account_status(self):
        """Update and calculate account status including daily metrics"""
        try:
            # Get current account summary
            current_info = self.oanda_client.get_account_summary()
            
            # Get saved account info at start of day
            day_start_info = self.memory.get_day_start_account_info()
            
            # If no day start info or new day, save current as day start
            current_date = datetime.datetime.now().date()
            last_date = day_start_info.get("date")
            
            if not last_date or last_date != current_date:
                day_start_info = {
                    "date": current_date,
                    "balance": current_info["balance"],
                    "high_balance": current_info["balance"]
                }
                self.memory.set_day_start_account_info(day_start_info)
            
            # Calculate daily P&L
            start_balance = day_start_info["balance"]
            current_balance = current_info["balance"]
            daily_pnl = current_balance - start_balance
            daily_pnl_pct = (daily_pnl / start_balance) * 100 if start_balance > 0 else 0
            
            # Calculate daily drawdown
            high_balance = day_start_info.get("high_balance", start_balance)
            if current_balance > high_balance:
                high_balance = current_balance
                # Update high balance in memory
                day_start_info["high_balance"] = high_balance
                self.memory.set_day_start_account_info(day_start_info)
                
            daily_drawdown = (high_balance - current_balance) / high_balance * 100 if high_balance > 0 else 0
            
            # Check if daily target or max drawdown reached
            target_reached = daily_pnl_pct >= self.daily_target_pct
            max_drawdown_reached = daily_drawdown >= self.max_daily_drawdown_pct
            
            # Combine into account status
            account_status = {
                "balance": current_balance,
                "currency": current_info["currency"],
                "daily_profit": daily_pnl,
                "daily_profit_pct": daily_pnl_pct,
                "daily_high": high_balance,
                "daily_drawdown": daily_drawdown,
                "target_reached": target_reached,
                "max_drawdown_reached": max_drawdown_reached,
                "unrealized_pl": current_info["unrealized_pl"],
                "margin_available": current_info.get("margin_available", 0)
            }
            
            return account_status
            
        except Exception as e:
            logger.error(f"Error updating account status: {e}")
            return {
                "balance": self.account_info["balance"],
                "currency": self.account_info["currency"],
                "daily_profit": 0,
                "daily_profit_pct": 0,
                "daily_drawdown": 0,
                "target_reached": False,
                "max_drawdown_reached": False,
                "error": str(e)
            }
    
    def open_position(self, direction, entry_price, stop_loss, take_profit, risk_percent):
        """Open a new position with specified parameters"""
        try:
            # Validate risk percentage
            risk_percent = min(risk_percent, self.max_risk_per_trade_pct)
            
            # Get current price data
            price_data = self.oanda_client.get_eur_usd_data(count=1, granularity="M1")
            if price_data.empty:
                self.last_error = "Failed to get current price data"
                return False
                
            current_price = price_data['close'].iloc[-1]
            
            # Determine units and direction
            account_balance = self.account_info["balance"]
            
            if direction == "BUY":
                # For buy orders, stop loss is below entry
                if stop_loss >= entry_price:
                    self.last_error = "Invalid stop loss for BUY order (should be below entry)"
                    return False
                    
                # Calculate position size
                units = self.oanda_client.calculate_position_size(account_balance, risk_percent, entry_price, stop_loss)
                
                # Create buy order (positive units)
                if abs(current_price - entry_price) / current_price < 0.0005:  # Within 0.05%
                    # Use market order for close prices
                    result = self.oanda_client.create_order("EUR_USD", units, None, stop_loss, take_profit)
                else:
                    # Use limit order for specific entry price
                    result = self.oanda_client.create_order("EUR_USD", units, entry_price, stop_loss, take_profit)
                    
            else:  # SELL
                # For sell orders, stop loss is above entry
                if stop_loss <= entry_price:
                    self.last_error = "Invalid stop loss for SELL order (should be above entry)"
                    return False
                    
                # Calculate position size
                units = self.oanda_client.calculate_position_size(account_balance, risk_percent, entry_price, stop_loss)
                
                # Negative units for sell orders
                units = -units
                
                # Create sell order (negative units)
                if abs(current_price - entry_price) / current_price < 0.0005:  # Within 0.05%
                    # Use market order for close prices
                    result = self.oanda_client.create_order("EUR_USD", units, None, stop_loss, take_profit)
                else:
                    # Use limit order for specific entry price
                    result = self.oanda_client.create_order("EUR_USD", units, entry_price, stop_loss, take_profit)
            
            # Check if order was created successfully
            if result.get("success", False):
                logger.info(f"Successfully opened {direction} position for EUR/USD")
                return True
            else:
                self.last_error = f"Failed to open position: {result.get('error', 'Unknown error')}"
                return False
                
        except Exception as e:
            self.last_error = f"Error opening position: {str(e)}"
            logger.error(f"Error opening position: {e}")
            return False
    
    def update_position(self, position_id=None, new_stop_loss=None, new_take_profit=None):
        """Update an existing position with new stop loss or take profit"""
        try:
            # If no position ID provided, get open positions
            if position_id is None:
                positions = self.oanda_client.get_open_positions()
                eur_usd_positions = [p for p in positions if p.get("instrument") == "EUR_USD"]
                
                if not eur_usd_positions:
                    self.last_error = "No open EUR/USD positions to update"
                    return False
                    
                # Use first position
                position_id = eur_usd_positions[0].get("id")
            
            # Update the position
            result = self.oanda_client.update_position_stops("EUR_USD", new_stop_loss, new_take_profit)
            
            # Check if update was successful
            if result.get("success", False):
                logger.info(f"Successfully updated position for EUR/USD")
                return True
            else:
                self.last_error = f"Failed to update position: {result.get('error', 'Unknown error')}"
                return False
                
        except Exception as e:
            self.last_error = f"Error updating position: {str(e)}"
            logger.error(f"Error updating position: {e}")
            return False
    
    def close_position(self, position_id=None):
        """Close an existing position"""
        try:
            # If no position ID provided, close all EUR/USD positions
            result = self.oanda_client.close_position("EUR_USD")
            
            # Check if close was successful
            if result.get("success", False):
                logger.info(f"Successfully closed position for EUR/USD")
                return True
            else:
                self.last_error = f"Failed to close position: {result.get('error', 'Unknown error')}"
                return False
                
        except Exception as e:
            self.last_error = f"Error closing position: {str(e)}"
            logger.error(f"Error closing position: {e}")
            return False
    
    def execute_decision(self, decision, price_data):
    """Execute a trading decision"""
    try:
        action = decision.get("action", "WAIT")
        
        if action == "WAIT":
            # Nothing to do
            return True
            
        elif action == "OPEN":
            # Handle opening a new position
            if "trade_details" not in decision:
                self.last_error = "Missing trade_details in decision"
                return False
                
            trade_details = decision["trade_details"]
            direction = trade_details.get("direction", "BUY")
            
            # Ensure we have proper price levels
            try:
                entry_price = float(trade_details.get("entry_price", 0))
                stop_loss = float(trade_details.get("stop_loss", 0))
                
                # Get take profit levels
                take_profit_levels = trade_details.get("take_profit", [])
                if isinstance(take_profit_levels, list) and len(take_profit_levels) > 0:
                    take_profit = float(take_profit_levels[0])
                else:
                    take_profit = float(take_profit_levels) if take_profit_levels else 0
            except (ValueError, TypeError):
                self.last_error = "Invalid price values in trade details"
                return False
            
            # Validate the trade
            if entry_price <= 0 or stop_loss <= 0 or take_profit <= 0:
                self.last_error = f"Invalid price levels: Entry={entry_price}, SL={stop_loss}, TP={take_profit}"
                return False
                
            # Calculate risk percentage (default to 1% if not specified)
            try:
                risk_percent = float(trade_details.get("risk_percent", 1.0))
                # Cap at 1% maximum
                risk_percent = min(risk_percent, 1.0)
            except (ValueError, TypeError):
                risk_percent = 1.0
            
            # Check current positions to avoid opening multiple
            positions = self.oanda_client.get_open_positions()
            if any(p.get('instrument') == 'EUR_USD' for p in positions):
                self.last_error = "Cannot open new position when one already exists"
                return False
            
            # Execute the trade (use your actual method for opening positions)
            if hasattr(self, 'open_position'):
                result = self.open_position(direction, entry_price, stop_loss, take_profit, risk_percent)
            else:
                # Fallback if open_position doesn't exist
                result = True
                # Add your opening position code here
            
            # Record the trade in memory if successful
            if result and hasattr(self, 'memory') and hasattr(self.memory, 'record_trade'):
                try:
                    self.memory.record_trade(
                        direction=direction,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_percent=risk_percent,
                        reasoning=trade_details.get("reasoning", "")
                    )
                except Exception as e:
                    # Don't fail the whole operation if recording fails
                    print(f"Warning: Failed to record trade: {e}")
                
            return result
            
        elif action == "UPDATE":
            # Handle updating an existing position
            
            # First check if we have a position to update
            positions = self.oanda_client.get_open_positions()
            eur_usd_positions = [p for p in positions if p.get("instrument") == "EUR_USD"]
            
            if not eur_usd_positions:
                self.last_error = "No EUR/USD positions found to update"
                return False
            
            # Get adjustment details
            position_management = decision.get("position_management", {})
            position_adjustment = position_management.get("position_adjustment", {})
            
            # Get new stop loss and take profit levels
            try:
                new_stop_loss = None
                if position_adjustment.get("adjust_stop_loss") is not None:
                    new_stop_loss = float(position_adjustment.get("adjust_stop_loss"))
                
                new_take_profit = None
                take_profit_adj = position_adjustment.get("adjust_take_profit")
                if take_profit_adj is not None:
                    if isinstance(take_profit_adj, list) and len(take_profit_adj) > 0:
                        new_take_profit = float(take_profit_adj[0])
                    else:
                        new_take_profit = float(take_profit_adj)
            except (ValueError, TypeError):
                self.last_error = "Invalid price values in position adjustment"
                return False
            
            # Check if we have adjustments to make
            if new_stop_loss is None and new_take_profit is None:
                self.last_error = "No stop loss or take profit adjustments specified"
                return False
            
            # Execute the update (use your actual method for updating positions)
            if hasattr(self, 'update_position'):
                result = self.update_position(
                    position_id=eur_usd_positions[0].get("id"),
                    new_stop_loss=new_stop_loss,
                    new_take_profit=new_take_profit
                )
            else:
                # Fallback if update_position doesn't exist
                result = True
                # Add your position update code here
            
            return result
            
        elif action == "CLOSE":
            # Handle closing an existing position
            
            # First check if we have a position to close
            positions = self.oanda_client.get_open_positions()
            eur_usd_positions = [p for p in positions if p.get("instrument") == "EUR_USD"]
            
            if not eur_usd_positions:
                self.last_error = "No EUR/USD positions found to close"
                return False
            
            # Execute the close (use your actual method for closing positions)
            if hasattr(self, 'close_position'):
                result = self.close_position(position_id=eur_usd_positions[0].get("id"))
            else:
                # Fallback if close_position doesn't exist
                result = True
                # Add your position closing code here
            
            # Record the position closure in memory if successful
            if result and hasattr(self, 'memory') and hasattr(self.memory, 'update_last_trade'):
                try:
                    self.memory.update_last_trade(is_closed=True)
                except Exception as e:
                    # Don't fail the whole operation if recording fails
                    print(f"Warning: Failed to update trade record: {e}")
            
            return result
            
        else:
            self.last_error = f"Unknown action: {action}"
            return False
            
    except Exception as e:
        self.last_error = f"Error executing decision: {str(e)}"
        return False
    
    def run_backtest(self, strategy_params, instrument="EUR_USD", days=30):
        """Run a backtest for a strategy with given parameters"""
        try:
            # Create a strategy instance based on params
            from brain import LLMGeneratedStrategy
            
            # Get strategy details
            strategy_name = strategy_params.get("name", "LLM Strategy")
            strategy_code = strategy_params.get("code", "")
            parameters = strategy_params.get("parameters", {})
            
            # Create strategy instance
            strategy = LLMGeneratedStrategy(strategy_name, parameters, strategy_code)
            
            # Define date range
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=days)
            
            # Run backtest
            result = self.backtester.run_backtest(
                strategy=strategy,
                instrument=instrument,
                start_date=start_date,
                end_date=end_date,
                timeframe="H1",
                initial_balance=10000.0,
                risk_per_trade=0.02  # 2% risk per trade for backtest
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return None