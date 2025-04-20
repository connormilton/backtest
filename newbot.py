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
import openai
import sys
from typing import Dict, List, Any, Optional, Tuple, Union
from dotenv import load_dotenv

# Import prompts from separate file
from prompts import SYSTEM_PROMPT, MARKET_ANALYSIS_PROMPT, SYSTEM_REVIEW_PROMPT

# Import backtesting engine components
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest_engine import (
    DataProvider, Strategy, Backtester, BacktestResult, TradeResult,
    CSVDataProvider, PolygonDataProvider
)

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LLM_Trader")

# Create output directories
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("strategies", exist_ok=True)

# API credentials from environment
OANDA_API_TOKEN = os.getenv("OANDA_API_TOKEN")
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
            return self._execute_open_action(decision, price_data)
        elif action == "UPDATE":
            return self._execute_update_action(decision)
        elif action == "CREATE":
            return self._execute_create_action(decision, price_data)
        
        # Implement early exit logic if positions need to be closed based on conditions
        if "exit_strategy" in decision and action != "OPEN":
            return self._execute_exit_strategy(decision)
            
        return False
    
    def _execute_open_action(self, decision, price_data):
        """Handle OPEN action - create new trading positions"""
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
        
        # Validate the trade with backtesting before proceeding
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
    
    def _execute_update_action(self, decision):
        """Handle UPDATE action - modify existing positions"""
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
        
        return False
    
    def _execute_create_action(self, decision, price_data):
        """Handle CREATE action - generate and validate new strategies"""
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
    
    def _execute_exit_strategy(self, decision):
        """Handle exit strategy - close positions based on conditions"""
        positions = self.oanda_client.get_open_positions()
        eur_usd_positions = [p for p in positions if p.get("instrument") == "EUR_USD"]
        
        if eur_usd_positions and "early_exit_conditions" in decision.get("exit_strategy", {}):
            logger.info(f"Evaluating early exit conditions")
            # Implementation to evaluate conditions and exit positions
            # This would be expanded based on the specific exit logic
            return True
        
        return False

# --- LLM Trading Brain ---
class LLMTradingBrain:
    """LLM-powered trading decision maker with self-improvement capabilities"""
    
    def __init__(self, memory, backtester):
        """Initialize the LLM trading brain"""
        self.memory = memory
        self.backtester = backtester
        openai.api_key = OPENAI_API_KEY
        
        # Initial system prompt (from imported SYSTEM_PROMPT)
        self.system_prompt = SYSTEM_PROMPT

        # Load the most recent successful strategy
        self.current_strategy = memory.load_strategy()
    
    def analyze_market(self, price_data, account_data, positions, market_data=None):
        """Analyze market and make trading decisions with comprehensive data"""
        # Extract recent price data from main timeframe
        recent_candles = price_data.tail(50).to_dict('records')
        simplified_candles = [
            {
                "time": str(candle["datetime"]) if "datetime" in candle else (candle["time"][-8:] if "time" in candle else ""),
                "open": round(candle["open"], 5),
                "high": round(candle["high"], 5),
                "low": round(candle["low"], 5),
                "close": round(candle["close"], 5)
            }
            for candle in recent_candles[-10:] # Last 10 candles
        ]
        
        # Get recent trades for context
        recent_trades = self.memory.get_recent_trades(5)
        
        # Calculate win rate
        win_rate = 0
        if self.memory.memory["trade_count"] > 0:
            win_rate = self.memory.memory["win_count"] / self.memory.memory["trade_count"] * 100
        
        # Ensure market_data is a dict
        if market_data is None:
            market_data = {}
            
        # Extract data components for prompt
        multi_timeframe_data = market_data.get("multi_timeframe", {})
        technical_indicators = market_data.get("technical_indicators", {})
        intermarket_data = market_data.get("intermarket", {})
        economic_data = market_data.get("economic", {})
        sentiment_data = market_data.get("sentiment", {})
        
        # Get current strategy info
        strategy_info = {
            "name": self.current_strategy.name,
            "parameters": self.current_strategy.parameters
        }
        
        # Build the prompt using the template from imported MARKET_ANALYSIS_PROMPT
        user_prompt = MARKET_ANALYSIS_PROMPT.format(
            balance=account_data.get('balance'),
            currency=account_data.get('currency'),
            safety_level=self.memory.memory['safety_level'],
            daily_profit_pct=self.memory.memory['daily_profit_pct'],
            win_rate=win_rate,
            win_count=self.memory.memory['win_count'],
            trade_count=self.memory.memory['trade_count'],
            open_positions=len([p for p in positions if p.get('instrument') == 'EUR_USD']),
            price_data=json.dumps(simplified_candles, indent=2),
            technical_indicators=json.dumps(technical_indicators, indent=2),
            multi_timeframe_data=json.dumps(multi_timeframe_data, indent=2),
            intermarket_data=json.dumps(intermarket_data, indent=2),
            economic_data=json.dumps(economic_data, indent=2),
            sentiment_data=json.dumps(sentiment_data, indent=2),
            recent_trades=json.dumps(recent_trades[:3], indent=2),
            strategy_info=json.dumps(strategy_info, indent=2),
            strategy_weights=json.dumps(self.memory.memory['strategy_weights'], indent=2)
        )
        
        # Call LLM API
        try:
            logger.info("Calling LLM API for market analysis")
            completion = openai.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
                max_tokens=4000  # Limit token count to avoid oversized responses
            )
            
            # Parse response with error handling
            try:
                result = json.loads(completion.choices[0].message.content)
                logger.info(f"LLM response received: {result.get('action', 'unknown action')}")
                return result
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON parsing error: {json_err}")
                # Return a safe default response
                return {"error": f"JSON parsing error: {str(json_err)}", "action": "WAIT"}
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            return {"error": str(e), "action": "WAIT"}
    
    def validate_strategy(self, strategy_details, price_data, risk_level=0.02, 
                         validation_period=30, confidence_threshold=MIN_CONFIDENCE_THRESHOLD):
        """Validate a strategy by running a backtest before deploying"""
        logger.info(f"Validating strategy: {strategy_details.get('name', 'Unnamed')}")
        
        try:
            # Create a dynamic strategy instance
            strategy = LLMGeneratedStrategy(
                name=strategy_details.get("name", "LLM Generated Strategy"),
                parameters=strategy_details.get("parameters", {}),
                strategy_code=strategy_details.get("code", "")
            )
            
            # Calculate validation period
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=validation_period)
            
            # Run backtest
            result = self.backtester.run_backtest(
                strategy=strategy,
                instrument="EUR/USD",
                start_date=start_date,
                end_date=end_date,
                timeframe="H1",  # Using hourly data for more granular testing
                initial_balance=10000.0,
                commission=0.0001,
                slippage=0.0001,
                risk_per_trade=risk_level,
                enable_stop_loss=True,
                enable_take_profit=True
            )
            
            if not result:
                logger.warning("Backtest returned no results for strategy validation")
                return {
                    "valid": False,
                    "confidence": 0.0,
                    "reason": "Backtest failed to produce results"
                }
            
            # Calculate confidence score based on multiple factors
            # (each on a scale of 0-1)
            factors = {
                "win_rate": min(1.0, result.win_rate / 100),  # 0-1 based on win rate (%)
                "sharpe": min(1.0, max(0, result.sharpe_ratio / 3)),  # 0-1 based on Sharpe (3+ is excellent)
                "profit_factor": min(1.0, result.profit_factor / 3 if result.profit_factor < float('inf') else 0.9),  # 0-1 based on profit factor
                "drawdown": 1 - min(1.0, result.max_drawdown_pct / 20),  # 0-1 inverted (lower drawdown is better)
                "trades": min(1.0, result.total_trades / 10)  # 0-1 based on number of trades (min 10 for full confidence)
            }
            
            # Calculate confidence score (weighted average)
            weights = {
                "win_rate": 0.30,  # 30% weight on win rate
                "sharpe": 0.25,    # 25% weight on risk-adjusted return
                "profit_factor": 0.20,  # 20% weight on profit factor
                "drawdown": 0.15,  # 15% weight on max drawdown
                "trades": 0.10     # 10% weight on number of trades
            }
            
            confidence = sum(factors[k] * weights[k] for k in factors)
            
            # Determine if strategy passes validation
            valid = (
                confidence >= confidence_threshold and 
                result.total_trades >= 5 and 
                result.profit_factor > 1.0 and
                result.win_rate >= 45
            )
            
            # Compile validation report
            validation_result = {
                "valid": valid,
                "confidence": confidence,
                "backtest_result": {
                    "total_return_pct": result.total_return_pct,
                    "win_rate": result.win_rate,
                    "total_trades": result.total_trades,
                    "profit_factor": result.profit_factor,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown_pct": result.max_drawdown_pct
                },
                "factor_scores": factors,
                "reason": "Strategy passed validation" if valid else "Strategy failed validation criteria"
            }
            
            # If valid, save the strategy for future use
            if valid:
                self.memory.log_strategy(strategy)
                self.current_strategy = strategy
            
            logger.info(f"Strategy validation completed with confidence {confidence:.2f}, Valid: {valid}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Error in strategy validation: {str(e)}")
            return {
                "valid": False,
                "confidence": 0.0,
                "reason": f"Error during validation: {str(e)}"
            }
    
    def validate_trade(self, trade_details, price_data):
        """Validate a trade by running a quick backtest with the current strategy"""
        logger.info(f"Validating trade: {trade_details.get('direction', 'Unknown')} at {trade_details.get('entry_price', 'Unknown')}")
        
        try:
            # Calculate validation period (last 30 days)
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=30)
            
            # Run backtest with current strategy
            result = self.backtester.run_backtest(
                strategy=self.current_strategy,
                instrument="EUR/USD",
                start_date=start_date,
                end_date=end_date,
                timeframe="H1",
                initial_balance=10000.0,
                commission=0.0001,
                slippage=0.0001,
                risk_per_trade=float(trade_details.get("risk_percent", 2)) / 100,
                enable_stop_loss=True,
                enable_take_profit=True
            )
            
            if not result:
                logger.warning("Backtest returned no results for trade validation")
                return {
                    "valid": False,
                    "confidence": 0.0,
                    "reason": "Backtest failed to produce results"
                }
            
            # Find similar trade setups in the backtest
            similar_trades = []
            
            # Current price and direction
            current_price = float(price_data.iloc[-1]["close"])
            direction = trade_details.get("direction", "BUY")
            
            for trade in result.trades:
                # Check if trade direction matches
                trade_direction = "BUY" if trade.direction == "BUY" else "SELL"
                if trade_direction != direction:
                    continue
                
                # Calculate how similar the entry price is (within 1%)
                entry_similarity = abs(trade.entry_price - current_price) / current_price
                if entry_similarity > 0.01:
                    continue
                
                # Check indicator similarity (would require more data to implement)
                # For now, just add the matching directional trades
                similar_trades.append(trade)
            
            # Calculate success rate of similar setups
            similar_count = len(similar_trades)
            winning_count = sum(1 for trade in similar_trades if trade.pnl > 0)
            similar_win_rate = winning_count / similar_count if similar_count > 0 else 0
            
            # Calculate risk-reward ratio
            entry_price = float(trade_details.get("entry_price", current_price))
            stop_loss = float(trade_details.get("stop_loss", 0))
            take_profits = trade_details.get("take_profit", [])
            
            # Ensure take_profit is a list
            if not isinstance(take_profits, list):
                take_profits = [float(take_profits)]
            else:
                take_profits = [float(tp) for tp in take_profits if tp]
            
            # Calculate risk and average reward
            if direction == "BUY":
                risk = entry_price - stop_loss if stop_loss > 0 else 0
                rewards = [(tp - entry_price) for tp in take_profits if tp > entry_price]
            else:  # SELL
                risk = stop_loss - entry_price if stop_loss > 0 else 0
                rewards = [(entry_price - tp) for tp in take_profits if tp < entry_price]
            
            avg_reward = sum(rewards) / len(rewards) if rewards else 0
            risk_reward_ratio = avg_reward / risk if risk > 0 else 0
            
            # Calculate overall confidence
            factors = {
                "backtest_win_rate": min(1.0, result.win_rate / 100),  # 0-1 based on backtest win rate
                "similar_win_rate": min(1.0, similar_win_rate),  # 0-1 based on similar setups
                "risk_reward": min(1.0, risk_reward_ratio / 2),  # 0-1 based on R:R (2+ is excellent)
                "sample_size": min(1.0, similar_count / 5)  # 0-1 based on number of similar trades
            }
            
            # Calculate confidence score (weighted average)
            weights = {
                "backtest_win_rate": 0.30,
                "similar_win_rate": 0.40,
                "risk_reward": 0.20,
                "sample_size": 0.10
            }
            
            confidence = sum(factors[k] * weights[k] for k in factors)
            
            # Determine if trade passes validation
            valid = (
                confidence >= MIN_CONFIDENCE_THRESHOLD and
                risk_reward_ratio >= 1.5 and
                (similar_count >= 3 or result.win_rate >= 55)
            )
            
            validation_result = {
                "valid": valid,
                "confidence": confidence,
                "backtest_summary": {
                    "win_rate": result.win_rate,
                    "total_trades": result.total_trades,
                    "profit_factor": result.profit_factor
                },
                "similar_setups": {
                    "count": similar_count,
                    "win_rate": similar_win_rate * 100
                },
                "risk_reward": {
                    "risk": risk,
                    "avg_reward": avg_reward,
                    "ratio": risk_reward_ratio
                },
                "factor_scores": factors,
                "reason": "Trade setup passed validation" if valid else "Trade setup failed validation criteria"
            }
            
            logger.info(f"Trade validation completed with confidence {confidence:.2f}, Valid: {valid}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Error in trade validation: {str(e)}")
            return {
                "valid": False,
                "confidence": 0.0,
                "reason": f"Error during validation: {str(e)}"
            }
    
    def review_and_evolve(self, account_data):
        """Review performance and evolve trading approach"""
        # Get all recent trades
        recent_trades = self.memory.get_recent_trades(20)
        
        # Calculate performance metrics
        win_rate = 0
        if self.memory.memory["trade_count"] > 0:
            win_rate = self.memory.memory["win_count"] / self.memory.memory["trade_count"] * 100
        
        # Calculate daily P&L
        daily_profit_pct = self.memory.memory["daily_profit_pct"]
        
        # Get saved strategies
        saved_strategies = self.memory.get_saved_strategies()
        
        # Build the review prompt using the template from imported SYSTEM_REVIEW_PROMPT
        review_prompt = SYSTEM_REVIEW_PROMPT.format(
            balance=account_data.get('balance'),
            currency=account_data.get('currency'),
            daily_profit_pct=daily_profit_pct,
            win_rate=win_rate,
            win_count=self.memory.memory['win_count'],
            trade_count=self.memory.memory['trade_count'],
            safety_level=self.memory.memory['safety_level'],
            strategy_weights=json.dumps(self.memory.memory['strategy_weights'], indent=2),
            system_prompt=self.system_prompt,
            recent_trades=json.dumps(recent_trades, indent=2),
            saved_strategies=json.dumps([{'name': s['name'], 'parameters': s['parameters']} for s in saved_strategies], indent=2)
        )
        
        # Call LLM API for review
        try:
            logger.info("Calling LLM API for system evolution review")
            completion = openai.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert trading system reviewer who improves a self-evolving forex trading AI."},
                    {"role": "user", "content": review_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
                max_tokens=4000  # Limit token count to avoid oversized responses
            )
            
            # Parse response with error handling
            try:
                result = json.loads(completion.choices[0].message.content)
                
                # Update system with recommendations
                if "improved_prompt" in result and result["improved_prompt"]:
                    self.system_prompt = result["improved_prompt"]
                    logger.info("Updated system prompt based on review")
                
                # Parse and convert strategy weights to floats
                if "strategy_weights" in result:
                    try:
                        strategy_weights = {}
                        for strategy, weight in result["strategy_weights"].items():
                            if isinstance(weight, (int, float)):
                                strategy_weights[strategy] = float(weight)
                            elif isinstance(weight, str):
                                # Try to convert string to float
                                try:
                                    strategy_weights[strategy] = float(weight.replace("float", "").strip())
                                except:
                                    strategy_weights[strategy] = 1.0  # Default
                            else:
                                strategy_weights[strategy] = 1.0  # Default
                        
                        # If valid weights were found, use them
                        if strategy_weights:
                            result["strategy_weights"] = strategy_weights
                            logger.info(f"Updated strategy weights: {strategy_weights}")
                    except Exception as e:
                        logger.error(f"Error converting strategy weights: {e}")
                
                # Log the review
                self.memory.log_review({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "review": result["performance_analysis"],
                    "strategy_weights": result.get("strategy_weights", self.memory.memory["strategy_weights"]),
                    "prompt_version": self.system_prompt,
                    "reason": result.get("reasoning", "Scheduled review")
                })
                
                logger.info("Completed system evolution review")
                return result
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON parsing error in review: {json_err}")
                return {"error": f"JSON parsing error: {str(json_err)}"}
                
        except Exception as e:
            logger.error(f"Error in system evolution review: {e}")
            return {"error": str(e)}

# --- Data Providers ---
class OandaDataProvider(DataProvider):
    """Data provider for OANDA API for backtesting recent data"""
    
    def __init__(self, api_token: str, account_id: str, practice: bool = True):
        """Initialize OANDA data provider"""
        self.api_token = api_token
        self.account_id = account_id
        self.base_url = "https://api-fxpractice.oanda.com" if practice else "https://api-fxtrade.oanda.com"
        
        # Set headers for all requests
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
            # Build API endpoint
            endpoint = f"{self.base_url}/v3/instruments/{oanda_instrument}/candles"
            
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
            endpoint = f"{self.base_url}/v3/accounts/{self.account_id}/instruments"
            
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
            # Build API endpoint
            endpoint = f"{self.base_url}/v3/instruments/{oanda_instrument}/candles"
            
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
        """Initialize OANDA API client"""
        self.session = requests.Session()
        
        # Set base URL based on account type
        self.base_url = "https://api-fxpractice.oanda.com" if OANDA_PRACTICE else "https://api-fxtrade.oanda.com"
        
        # Set headers for all requests
        self.headers = {
            "Authorization": f"Bearer {OANDA_API_TOKEN}",
            "Content-Type": "application/json"
        }
        self.session.headers.update(self.headers)
        
        # Verify connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to OANDA API"""
        account = self.get_account()
        logger.info(f"Connected to OANDA. Balance: {account.get('balance')} {account.get('currency')}")
    
    def get_account(self):
        """Get account information"""
        try:
            response = self.session.get(f"{self.base_url}/v3/accounts/{OANDA_ACCOUNT_ID}/summary")
            response.raise_for_status()
            return response.json().get("account", {})
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            return {}
    
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
        """Get EUR/USD price data for a specific timeframe"""
        try:
            params = {
                "count": count,
                "granularity": granularity,
                "price": "M"
            }
            response = self.session.get(
                f"{self.base_url}/v3/instruments/EUR_USD/candles", 
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
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error getting EUR/USD data: {e}")
            return pd.DataFrame()
            
    def get_multi_timeframe_data(self):
        """Get EUR/USD data across multiple timeframes"""
        timeframes = {
            "M5": {"count": 100, "name": "5-minute"},
            "M15": {"count": 100, "name": "15-minute"},
            "H1": {"count": 100, "name": "1-hour"},
            "H4": {"count": 50, "name": "4-hour"},
            "D": {"count": 30, "name": "Daily"}
        }
        
        data = {}
        for tf, tf_info in timeframes.items():
            df = self.get_eur_usd_data(count=tf_info["count"], granularity=tf)
            if not df.empty:
                # Convert DataFrame to list of dicts for easier serialization
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
                indicators["support_resistance"] = {
                    "key_resistance_levels": [],
                    "key_support_levels": []
                }
            
            return indicators
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {}
    
    def get_open_positions(self):
        """Get open positions"""
        try:
            response = self.session.get(f"{self.base_url}/v3/accounts/{OANDA_ACCOUNT_ID}/openPositions")
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
            response = self.session.post(
                f"{self.base_url}/v3/accounts/{OANDA_ACCOUNT_ID}/orders",
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
            response = self.session.get(
                f"{self.base_url}/v3/accounts/{OANDA_ACCOUNT_ID}/trades?instrument=EUR_USD&state=OPEN"
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
                    update_response = self.session.put(
                        f"{self.base_url}/v3/accounts/{OANDA_ACCOUNT_ID}/trades/{trade_id}/orders",
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

# --- LLM Generated Strategy ---
class LLMGeneratedStrategy(Strategy):
    """Strategy dynamically generated by LLM with parameters"""
    
    def __init__(self, name="LLM Generated Strategy", parameters=None, strategy_code=None):
        """Initialize with strategy parameters and executable code"""
        super().__init__(name)
        
        # Set default parameters
        self.parameters = parameters or {}
        
        # Store strategy code
        self.strategy_code = strategy_code or """
def generate_signals(self, data):
    # Default implementation - SMA Crossover
    if data.empty:
        return data
    
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Calculate fast and slow SMAs
    fast_period = self.parameters.get('fast_period', 10)
    slow_period = self.parameters.get('slow_period', 30)
    
    df['fast_ma'] = df['close'].rolling(window=fast_period).mean()
    df['slow_ma'] = df['close'].rolling(window=slow_period).mean()
    
    # Initialize signal column
    df['signal'] = 0
    
    # Calculate crossover signals
    df['prev_fast_ma'] = df['fast_ma'].shift(1)
    df['prev_slow_ma'] = df['slow_ma'].shift(1)
    
    # Buy signal: fast MA crosses above slow MA
    buy_condition = (df['prev_fast_ma'] < df['prev_slow_ma']) & (df['fast_ma'] > df['slow_ma'])
    df.loc[buy_condition, 'signal'] = 1
    
    # Sell signal: fast MA crosses below slow MA
    sell_condition = (df['prev_fast_ma'] > df['prev_slow_ma']) & (df['fast_ma'] < df['slow_ma'])
    df.loc[sell_condition, 'signal'] = -1
    
    return df
"""
        
        # Create the function dynamically
        self._create_dynamic_method()
    
    def _create_dynamic_method(self):
        """Create generate_signals method dynamically from code string"""
        try:
            # Create a local namespace
            local_namespace = {}
            
            # Execute the code in the local namespace
            exec(self.strategy_code, globals(), local_namespace)
            
            # Bind the function to the instance
            self.generate_signals_impl = local_namespace.get('generate_signals')
            
            if not self.generate_signals_impl:
                logger.error("Strategy code did not define a generate_signals function")
                # Set default implementation
                self.generate_signals_impl = lambda self, data: data
                
        except Exception as e:
            logger.error(f"Error creating dynamic method: {str(e)}")
            # Set default implementation
            self.generate_signals_impl = lambda self, data: data
    
    def generate_signals(self, data):
        """Generate trading signals using dynamically created method"""
        try:
            # Call the dynamically created implementation
            return self.generate_signals_impl(self, data)
        except Exception as e:
            logger.error(f"Error in dynamic strategy: {str(e)}")
            # Return original data if there's an error
            return data
    
    def save_to_file(self, filename=None):
        """Save strategy to file for later reuse"""
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = self.name.replace(" ", "_").lower()
            filename = f"strategies/{safe_name}_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        strategy_data = {
            "name": self.name,
            "parameters": self.parameters,
            "strategy_code": self.strategy_code
        }
        
        with open(filename, 'w') as f:
            json.dump(strategy_data, f, indent=2)
        
        logger.info(f"Strategy saved to {filename}")
        return filename
    
    @staticmethod
    def load_from_file(filename):
        """Load strategy from file"""
        with open(filename, 'r') as f:
            strategy_data = json.load(f)
        
        return LLMGeneratedStrategy(
            name=strategy_data.get("name", "Loaded Strategy"),
            parameters=strategy_data.get("parameters", {}),
            strategy_code=strategy_data.get("strategy_code", "")
        )

# --- Memory System ---
class TradingMemory:
    """Maintains trading history and system state"""
    
    def __init__(self):
        """Initialize trading memory"""
        self.memory_file = "data/system_memory.json"
        self.trade_log_file = "data/trade_log.jsonl"
        self.review_log_file = "data/review_log.jsonl"
        self.strategies_dir = "strategies"
        
        # Ensure data directories exist
        os.makedirs("data", exist_ok=True)
        os.makedirs(self.strategies_dir, exist_ok=True)
        
        # Initialize or load system memory
        self.load_memory()
    
    def load_memory(self):
        """Load or initialize system memory"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r") as f:
                    self.memory = json.load(f)
            except:
                self.initialize_memory()
        else:
            self.initialize_memory()
    
    def initialize_memory(self):
        """Create initial memory structure"""
        self.memory = {
            "created": datetime.datetime.now().isoformat(),
            "last_updated": datetime.datetime.now().isoformat(),
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "safety_level": INITIAL_SAFETY_LEVEL,
            "daily_drawdown": 0.0,
            "daily_profit": 0.0,
            "daily_profit_pct": 0.0,
            "daily_high_balance": 0.0,
            "daily_starting_balance": 0.0,
            "total_risk_committed": 0.0,
            "prompt_versions": [],
            "strategy_weights": {
                "trend_following": 1.0,
                "breakout": 1.0,
                "mean_reversion": 1.0
            },
            "saved_strategies": []
        }
        self.save_memory()
    
    def save_memory(self):
        """Save system memory to disk"""
        self.memory["last_updated"] = datetime.datetime.now().isoformat()
        with open(self.memory_file, "w") as f:
            json.dump(self.memory, f, indent=2)
    
    def log_trade(self, trade_data):
        """Log a trade to the trade history"""
        # Update trade count
        self.memory["trade_count"] += 1
        
        # Update win/loss stats if applicable
        if trade_data.get("is_win"):
            self.memory["win_count"] += 1
            # Increase safety level after win
            self.memory["safety_level"] = min(0.05, self.memory["safety_level"] + INCREASE_FACTOR)
        elif trade_data.get("is_loss"):
            self.memory["loss_count"] += 1
        
        # Add timestamp if not provided
        if "timestamp" not in trade_data:
            trade_data["timestamp"] = datetime.datetime.now().isoformat()
        
        # Append to trade log
        with open(self.trade_log_file, "a") as f:
            f.write(json.dumps(trade_data) + "\n")
        
        # Save updated memory
        self.save_memory()
    
    def log_review(self, review_data):
        """Log a system review"""
        # Save the new prompt version if provided
        if "prompt_version" in review_data:
            self.memory["prompt_versions"].append({
                "timestamp": datetime.datetime.now().isoformat(),
                "content": review_data["prompt_version"],
                "reason": review_data.get("reason", "Regular review")
            })
        
        # Update strategy weights if provided
        if "strategy_weights" in review_data:
            self.memory["strategy_weights"] = review_data["strategy_weights"]
        
        # Append to review log
        with open(self.review_log_file, "a") as f:
            f.write(json.dumps(review_data) + "\n")
        
        # Save updated memory
        self.save_memory()
    
    def log_strategy(self, strategy):
        """Save a successful strategy to memory"""
        # Save strategy to file
        filename = strategy.save_to_file()
        
        # Add to memory
        if "saved_strategies" not in self.memory:
            self.memory["saved_strategies"] = []
            
        self.memory["saved_strategies"].append({
            "name": strategy.name,
            "timestamp": datetime.datetime.now().isoformat(),
            "filename": filename,
            "parameters": strategy.parameters
        })
        
        # Save updated memory
        self.save_memory()
    
    def get_recent_trades(self, limit=10):
        """Get recent trades from log"""
        trades = []
        try:
            if os.path.exists(self.trade_log_file):
                with open(self.trade_log_file, "r") as f:
                    for line in f:
                        trades.append(json.loads(line))
                
                # Sort by timestamp (newest first) and limit
                trades.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                return trades[:limit]
        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
        
        return trades
    
    def reset_daily_stats(self, account_balance):
        """Reset daily statistics"""
        self.memory["daily_drawdown"] = 0.0
        self.memory["daily_profit"] = 0.0
        self.memory["daily_profit_pct"] = 0.0
        self.memory["daily_high_balance"] = account_balance
        self.memory["daily_starting_balance"] = account_balance
        self.memory["total_risk_committed"] = 0.0
        self.save_memory()
    
    def get_saved_strategies(self, limit=5):
        """Get the most recent saved strategies"""
        if "saved_strategies" not in self.memory:
            return []
            
        strategies = self.memory["saved_strategies"]
        strategies.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return strategies[:limit]
    
    def load_strategy(self, strategy_name=None, strategy_file=None):
        """Load a strategy from memory or file"""
        if strategy_file:
            return LLMGeneratedStrategy.load_from_file(strategy_file)
        
        if strategy_name:
            strategies = self.memory.get("saved_strategies", [])
            for strategy in strategies:
                if strategy.get("name") == strategy_name and os.path.exists(strategy.get("filename", "")):
                    return LLMGeneratedStrategy.load_from_file(strategy.get("filename"))
        
        # Return most recent if not found
        strategies = self.get_saved_strategies(1)
        if strategies:
            strategy = strategies[0]
            if os.path.exists(strategy.get("filename", "")):
                return LLMGeneratedStrategy.load_from_file(strategy.get("filename"))
        
        # Return default if no saved strategies
        return LLMGeneratedStrategy()

# Main function - run when script is executed directly
if __name__ == "__main__":
    try:
        # Initialize trading bot
        bot = EURUSDTradingBot()
        
        # Main trading loop
        while True:
            try:
                # Update account status
                account_status = bot.update_account_status()
                
                # Check if daily limits reached
                if account_status["max_drawdown_reached"]:
                    logger.warning(f"Maximum daily drawdown reached: {account_status['daily_drawdown']:.2f}%. Stopping for the day.")
                    break
                    
                if account_status["target_reached"]:
                    logger.info(f"Daily target reached: {account_status['daily_profit_pct']:.2f}%. Stopping for the day.")
                    break
                
                # Get current price data
                price_data = bot.oanda_data_provider.get_latest_data("EUR/USD", count=100, timeframe="H1")
                
                if price_data.empty:
                    logger.error("Failed to get price data. Retrying in 5 minutes.")
                    time.sleep(300)
                    continue
                
                # Get open positions
                positions = bot.oanda_client.get_open_positions()
                
                # Prepare market data
                market_data = {
                    "multi_timeframe": bot.oanda_client.get_multi_timeframe_data(),
                    "technical_indicators": bot.oanda_client.get_technical_indicators(price_data),
                    "intermarket": bot.oanda_client.get_intermarket_data(),
                    "economic": bot.oanda_client.get_economic_calendar(),
                    "sentiment": bot.oanda_client.get_market_sentiment()
                }
                
                # Analyze market and get decision
                decision = bot.brain.analyze_market(
                    price_data=price_data,
                    account_data=account_status,
                    positions=positions,
                    market_data=market_data
                )
                
                # Execute decision
                if "action" in decision:
                    action = decision.get("action")
                    logger.info(f"Decision: {action}")
                    
                    # Execute the decision
                    result = bot.execute_decision(decision, price_data)
                    
                    if result:
                        logger.info(f"Executed {action} successfully")
                    else:
                        logger.warning(f"Failed to execute {action}")
                
                # Periodically evolve the system (e.g., every 24 hours)
                current_time = datetime.datetime.now()
                if current_time.hour == 0 and current_time.minute < 5:  # Around midnight
                    logger.info("Performing system evolution review")
                    bot.brain.review_and_evolve(account_status)
                
                # Sleep to avoid API rate limits
                time.sleep(300)  # 5 minutes between checks
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")