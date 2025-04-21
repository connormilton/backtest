#!/usr/bin/env python3
"""
Enhanced Trading Bot Brain Module with improved OpenAI integration
This version uses a more comprehensive analysis prompt while maintaining reliable JSON responses
and integrates directly with the backtesting engine
"""

import os
import json
import logging
import datetime
import pandas as pd
import numpy as np
import openai
import requests
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from enhanced_openai_prompt import ENHANCED_MARKET_ANALYSIS_PROMPT, ENHANCED_SYSTEM_PROMPT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Trading_Brain")

class EnhancedTradingBrain:
    """LLM-powered trading decision maker with enhanced reasoning and reliable JSON output"""
    
    def __init__(self, memory, backtester):
        """Initialize the LLM trading brain"""
        self.memory = memory
        self.backtester = backtester
        
        # Get API key from environment
        from bot import OPENAI_API_KEY
        openai.api_key = OPENAI_API_KEY
        
        # Use the enhanced system prompt
        self.system_prompt = ENHANCED_SYSTEM_PROMPT
        
        # Add risk management parameters
        self.max_total_risk_percent = 1.0  # Maximum 1% total risk
        self.max_open_positions = 1  # Only allow one position at a time
    
    def analyze_market(self, price_data, account_data, positions, market_data=None):
        """Analyze market data and generate trading signals with comprehensive reasoning"""
        try:
            # Default response if analysis fails
            default_action = {"action": "WAIT", "reason": "Analysis failed or insufficient data"}
            
            if price_data.empty:
                logger.warning("Empty price data provided for analysis")
                return default_action
            
            # Prepare memory metrics
            memory_stats = {
                "balance": account_data.get("balance", 0),
                "currency": "USD",
                "safety_level": self.memory.memory.get("safety_level", 0.01) if hasattr(self.memory, "memory") else 0.01,
                "daily_profit_pct": account_data.get("daily_profit_pct", 0),
                "win_rate": (self.memory.memory.get("win_count", 0) / self.memory.memory.get("trade_count", 1)) * 100 if hasattr(self.memory, "memory") and self.memory.memory.get("trade_count", 0) > 0 else 0,
                "win_count": self.memory.memory.get("win_count", 0) if hasattr(self.memory, "memory") else 0,
                "trade_count": self.memory.memory.get("trade_count", 0) if hasattr(self.memory, "memory") else 0
            }
            
            # Get recent trades
            recent_trades = []
            if hasattr(self.memory, "get_recent_trades"):
                recent_trades = self.memory.get_recent_trades(5)
            
            # Format recent trades for prompt
            recent_trades_formatted = []
            for trade in recent_trades:
                trade_str = f"Direction: {trade.get('direction', 'Unknown')}, "
                trade_str += f"Entry: {trade.get('entry_price', 'Unknown')}, "
                trade_str += f"Exit: {trade.get('exit_price', 'Unknown') if trade.get('exit_price') else 'Open'}, "
                trade_str += f"Result: {'Win' if trade.get('is_win') else 'Loss' if trade.get('is_loss') else 'Unknown/Open'}, "
                trade_str += f"Reasoning: {trade.get('reasoning', 'None provided')[:100]}..."
                recent_trades_formatted.append(trade_str)
            
            # Check for existing EUR_USD positions
            eur_usd_positions = [p for p in positions if p.get('instrument') == 'EUR_USD']
            has_existing_position = len(eur_usd_positions) > 0
            
            # Format open positions with explicit position status
            position_status = "POSITIONS: OPEN - " if has_existing_position else "POSITIONS: NONE - "
            if not positions or not has_existing_position:
                open_positions = position_status + "No open EUR/USD positions"
            else:
                position_details = ", ".join([
                    f"ID: {p.get('id', 'Unknown')}, Direction: {'BUY' if int(p.get('long', {}).get('units', 0)) > 0 else 'SELL'}, "
                    f"Units: {abs(int(p.get('long', {}).get('units', 0)) or int(p.get('short', {}).get('units', 0)))}, "
                    f"Unrealized P&L: {p.get('unrealizedPL', 'Unknown')}"
                    for p in positions if p.get('instrument') == 'EUR_USD'
                ])
                open_positions = position_status + position_details
            
            # Prepare price data summary - just include most recent candles
            price_sample = price_data.tail(10).to_string()
            
            # Calculate and format technical indicators
            self._calculate_indicators(price_data)
            
            technical_indicators = f"""
Last Close: {price_data['close'].iloc[-1]:.5f}
Daily Change: {(price_data['close'].iloc[-1] / price_data['close'].iloc[-2] - 1) * 100:.2f}%
ATR(14): {price_data['atr'].iloc[-1]:.5f}
RSI(14): {price_data['rsi'].iloc[-1]:.2f}
MACD: {price_data['macd'].iloc[-1]:.5f}, Signal: {price_data['macd_signal'].iloc[-1]:.5f}
Bollinger Bands: Upper: {price_data['bb_upper'].iloc[-1]:.5f}, Middle: {price_data['bb_middle'].iloc[-1]:.5f}, Lower: {price_data['bb_lower'].iloc[-1]:.5f}
20 SMA: {price_data['sma_20'].iloc[-1]:.5f}
50 SMA: {price_data['sma_50'].iloc[-1]:.5f}
200 SMA: {price_data['sma_200'].iloc[-1]:.5f}
Trend: {"Bullish" if price_data['sma_20'].iloc[-1] > price_data['sma_50'].iloc[-1] else "Bearish"}
Volume: {price_data['volume'].iloc[-1]}
"""
            
            # Format market data components
            intermarket_data = "Not available"
            if market_data and "intermarket" in market_data:
                intermarket_data = str(market_data["intermarket"])
            
            economic_data = "Not available"
            if market_data and "economic" in market_data:
                economic_data = str(market_data["economic"])
            
            sentiment_data = "Not available"
            if market_data and "sentiment" in market_data:
                sentiment_data = str(market_data["sentiment"])
            
            # Get saved strategies
            strategies = []
            if hasattr(self.memory, "get_saved_strategies"):
                strategies = self.memory.get_saved_strategies(3)
            strategy_info = "None" if not strategies else ", ".join([f"{s.get('name', 'Unknown')}" for s in strategies])
            
            # Add risk management guidance to the prompt
            risk_management_note = f"""
IMPORTANT RISK MANAGEMENT CONSTRAINTS:
- Maximum total account risk: {self.max_total_risk_percent}%
- Maximum open positions: {self.max_open_positions} (currently: {len(eur_usd_positions)})
"""
            if has_existing_position:
                risk_management_note += """
IMPORTANT: You ARE CURRENTLY MANAGING AN EXISTING POSITION. Your options are:
1. WAIT - Continue to monitor the position without changes
2. UPDATE - Adjust the stop loss or take profit levels
3. CLOSE - Close the position if conditions warrant
DO NOT suggest opening a new position until the existing one is closed.
"""
            else:
                risk_management_note += """
IMPORTANT: You DO NOT have any open positions currently. Your options are:
1. WAIT - If market conditions don't warrant a trade
2. OPEN - Open a new position if a high-conviction setup exists (ensure risk <= 1% of account)
UPDATE and CLOSE actions are NOT VALID when no position exists.
"""
            
            # Get strategy weights (if they exist)
            strategy_weights = {}
            if hasattr(self.memory, "memory") and "strategy_weights" in self.memory.memory:
                strategy_weights = self.memory.memory.get("strategy_weights", {})
            
            # Create prompt with enhanced format
            prompt = ENHANCED_MARKET_ANALYSIS_PROMPT.format(
                balance=memory_stats["balance"],
                currency=memory_stats["currency"],
                safety_level=memory_stats["safety_level"],
                daily_profit_pct=memory_stats["daily_profit_pct"],
                win_rate=memory_stats["win_rate"],
                win_count=memory_stats["win_count"],
                trade_count=memory_stats["trade_count"],
                open_positions=open_positions,
                price_data=price_sample,
                technical_indicators=technical_indicators,
                intermarket_data=intermarket_data,
                economic_data=economic_data,
                sentiment_data=sentiment_data,
                recent_trades="\n".join(recent_trades_formatted),
                strategy_info=strategy_info,
                strategy_weights=str(strategy_weights)
            )
            
            # Add risk management note to the prompt
            prompt = prompt + "\n\n" + risk_management_note
            
            # Call OpenAI API using direct client
            try:
                # Import the OpenAI client directly
                from openai import OpenAI
                
                # Initialize client
                client = OpenAI(api_key=openai.api_key)
                
                # Log that we're about to make the API call
                logger.info("Calling OpenAI API with enhanced prompt...")
                
                # Make the API call
                response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=4000
                )
                
                # Extract the response text
                decision_text = response.choices[0].message.content
                
                # Log the first part of the response for diagnostic purposes
                logger.info(f"Raw LLM response (first 200 chars): {decision_text[:200]}...")
                
                # Extract and parse the JSON part of the response
                decision_json = self._extract_json_from_text(decision_text)
                
                if decision_json:
                    # Extract the action field or set default
                    action = decision_json.get("action", "WAIT")
                    if not action:
                        action = "WAIT"
                    
                    # Apply risk management constraints
                    decision_json = self._apply_risk_management(decision_json, has_existing_position, account_data)
                    
                    # Ensure action is in the result
                    decision_json["action"] = decision_json.get("action", "WAIT")
                    
                    # Include the analysis part of the response (non-JSON part)
                    # This captures the detailed reasoning before the JSON structure
                    json_start = decision_text.find('{')
                    if json_start > 0:
                        detailed_analysis = decision_text[:json_start].strip()
                        decision_json["detailed_analysis"] = detailed_analysis
                    
                    # If the action is OPEN, run backtesting validation
                    if decision_json["action"] == "OPEN" and "trade_details" in decision_json:
                        logger.info("Validating trade through backtesting...")
                        
                        # Extract backtesting parameters from the response
                        backtesting_params = decision_json.get("backtesting_parameters", {})
                        lookback_period = backtesting_params.get("lookback_period", 30)
                        
                        # Extract validation metrics
                        validation_metrics = backtesting_params.get("validation_metrics", {})
                        min_win_rate = validation_metrics.get("min_win_rate", 50)
                        min_profit_factor = validation_metrics.get("min_profit_factor", 1.5)
                        max_drawdown = validation_metrics.get("max_drawdown", 20)
                        
                        # Convert min_win_rate from percentage string to float if needed
                        if isinstance(min_win_rate, str):
                            min_win_rate = float(min_win_rate.replace('%', ''))
                            
                        # Run backtest to validate the trade
                        backtest_result = self._run_backtest_validation(
                            decision_json["trade_details"],
                            price_data,
                            lookback_period=lookback_period,
                            min_win_rate=min_win_rate,
                            min_profit_factor=min_profit_factor,
                            max_drawdown=max_drawdown
                        )
                        
                        # Include backtesting results in the decision
                        decision_json["backtest_validation"] = backtest_result
                        
                        # If backtesting failed, update the action
                        if not backtest_result.get("valid", True):
                            logger.warning(f"Backtest validation failed: {backtest_result.get('reason')}")
                            
                            # Modify action only if validation wasn't successful
                            decision_json["action"] = "WAIT"
                            decision_json["reason"] = f"Backtest validation failed: {backtest_result.get('reason')}"
                    
                    logger.info(f"Final decision action: {decision_json.get('action')}")
                    return decision_json
                else:
                    logger.warning("Failed to extract JSON from response")
                    # If JSON extraction fails, use fallback parsing
                    
                    # Apply position management constraints even to fallback parsing
                    if has_existing_position and ("BUY" in decision_text or "SELL" in decision_text):
                        logger.warning("Preventing new position when one already exists")
                        return {
                            "action": "WAIT", 
                            "reason": "Already have an open position. Cannot open another until existing position is closed.",
                            "detailed_analysis": decision_text[:1000]
                        }
                    
                    # Ensure we don't try to update or close when no position exists
                    if not has_existing_position and ("UPDATE" in decision_text or "CLOSE" in decision_text):
                        logger.warning("Preventing update/close when no position exists")
                        return {
                            "action": "WAIT", 
                            "reason": "No open position to update or close.",
                            "detailed_analysis": decision_text[:1000]
                        }
                    
                    if "BUY" in decision_text:
                        return {
                            "action": "OPEN",
                            "trade_details": {
                                "direction": "BUY",
                                "reasoning": "Extracted from non-JSON response",
                                "risk_percent": self.max_total_risk_percent  # Enforce max risk
                            },
                            "detailed_analysis": decision_text[:1000]  # Include full text as analysis
                        }
                    elif "SELL" in decision_text:
                        return {
                            "action": "OPEN", 
                            "trade_details": {
                                "direction": "SELL",
                                "reasoning": "Extracted from non-JSON response",
                                "risk_percent": self.max_total_risk_percent  # Enforce max risk
                            },
                            "detailed_analysis": decision_text[:1000]
                        }
                    else:
                        return {
                            "action": "WAIT", 
                            "reason": "Could not parse response as JSON",
                            "detailed_analysis": decision_text[:1000]
                        }
                    
            except Exception as e:
                logger.error(f"Error calling OpenAI API: {str(e)}")
                return {"action": "WAIT", "reason": f"OpenAI API error: {str(e)}"}
                
        except Exception as e:
            logger.error(f"Error analyzing market: {e}")
            return {"action": "WAIT", "reason": f"Analysis error: {str(e)}"}
    
    def _apply_risk_management(self, decision, has_existing_position, account_data):
        """Apply risk management constraints to the decision"""
        action = decision.get("action", "WAIT")
        
        # Check for logical conflicts in action vs. position state
        if not has_existing_position and action in ["UPDATE", "CLOSE"]:
            # Can't update or close non-existent positions
            logger.warning(f"Action {action} not valid with no open positions. Changing to WAIT.")
            decision["action"] = "WAIT"
            decision["reason"] = f"Cannot {action.lower()} a position that doesn't exist"
            return decision
        
        # Check if we already have a position
        if has_existing_position and action == "OPEN":
            # Change action to WAIT if trying to open a new position when one exists
            logger.warning("Already have an open position. Changing action from OPEN to WAIT.")
            decision["action"] = "WAIT"
            decision["reason"] = "Already have an open position. Cannot open another until existing position is closed."
            return decision
        
        # Enforce maximum risk for OPEN actions
        if action == "OPEN" and "trade_details" in decision:
            # Get the risk percentage from the decision
            risk_percent = decision["trade_details"].get("risk_percent", 0)
            
            # Try to convert to float if it's a string
            if isinstance(risk_percent, str):
                try:
                    risk_percent = float(risk_percent.replace('%', ''))
                except:
                    risk_percent = 0
            
            # If risk is higher than max allowed, cap it
            if risk_percent > self.max_total_risk_percent:
                logger.warning(f"Reducing risk from {risk_percent}% to {self.max_total_risk_percent}%")
                decision["trade_details"]["risk_percent"] = self.max_total_risk_percent
                decision["trade_details"]["original_risk_percent"] = risk_percent
            
            # If risk is too low or invalid, set it to the maximum allowed
            if risk_percent <= 0:
                logger.warning(f"Invalid risk {risk_percent}%. Setting to {self.max_total_risk_percent}%")
                decision["trade_details"]["risk_percent"] = self.max_total_risk_percent
        
        return decision
    
    def _calculate_indicators(self, data):
        """Calculate technical indicators for price data"""
        # Make sure we have enough data
        if len(data) < 200:
            # Pad with NaN if needed
            logger.warning("Insufficient data for all indicators, some values will be NaN")
        
        # Simple Moving Averages
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        data['sma_200'] = data['close'].rolling(window=200).mean()
        
        # RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
        data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_hist'] = data['macd'] - data['macd_signal']
        
        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=20).mean()
        data['bb_std'] = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + (2 * data['bb_std'])
        data['bb_lower'] = data['bb_middle'] - (2 * data['bb_std'])
        
        # ATR (Average True Range)
        high_low = data['high'] - data['low']
        high_close = (data['high'] - data['close'].shift()).abs()
        low_close = (data['low'] - data['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        data['atr'] = true_range.rolling(window=14).mean()
    
    def _extract_json_from_text(self, text):
        """Extract and parse JSON from the response text"""
        try:
            # Find the JSON object in the text (between { and })
            pattern = r'({[\s\S]*})'
            match = re.search(pattern, text)
            
            if not match:
                logger.warning("No JSON object found in response")
                return None
                
            json_str = match.group(1)
            
            # Try to parse the JSON
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Initial JSON parsing failed: {e}")
                
                # Try to fix common JSON formatting issues
                clean_json = json_str
                
                # Remove trailing commas before closing braces/brackets
                clean_json = re.sub(r',\s*}', '}', clean_json)
                clean_json = re.sub(r',\s*]', ']', clean_json)
                
                # Fix newlines before quotes
                clean_json = re.sub(r'\n\s*"', '"', clean_json)
                
                # Ensure property names are quoted
                clean_json = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', clean_json)
                
                # Try again with cleaned JSON
                try:
                    return json.loads(clean_json)
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON even after cleaning")
                    return None
                    
        except Exception as e:
            logger.error(f"Error extracting JSON: {e}")
            return None
            
    def _run_backtest_validation(self, trade_details, price_data, lookback_period=30, 
                                min_win_rate=50, min_profit_factor=1.5, max_drawdown=20):
        """Run backtesting to validate a trade before execution"""
        try:
            # Extract trade parameters
            direction = trade_details.get("direction", "BUY")
            entry_price = float(trade_details.get("entry_price", 0))
            stop_loss = float(trade_details.get("stop_loss", 0))
            
            # Get first take profit level
            take_profit_levels = trade_details.get("take_profit", [])
            if isinstance(take_profit_levels, list) and len(take_profit_levels) > 0:
                take_profit = float(take_profit_levels[0])
            else:
                take_profit = float(take_profit_levels) if take_profit_levels else 0
            
            risk_percent = float(trade_details.get("risk_percent", self.max_total_risk_percent))
            strategy_type = trade_details.get("strategy", "trend_following")
            
            # Validate that stop loss is not too close to entry
            current_price = price_data['close'].iloc[-1]
            min_stop_distance = current_price * 0.001  # Min 0.1% distance
            
            if direction == "BUY" and entry_price - stop_loss < min_stop_distance:
                return {
                    "valid": False,
                    "reason": "Stop loss too close to entry price",
                    "metrics": {}
                }
            elif direction == "SELL" and stop_loss - entry_price < min_stop_distance:
                return {
                    "valid": False,
                    "reason": "Stop loss too close to entry price",
                    "metrics": {}
                }
            
            # Create appropriate strategy for backtesting based on the strategy type
            strategy = self._create_strategy_for_backtest(
                strategy_type, 
                direction, 
                entry_price, 
                stop_loss, 
                take_profit
            )
            
            # Calculate date range
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=lookback_period)
            
            # Run backtest
            result = self.backtester.run_backtest(
                strategy=strategy,
                instrument="EUR/USD",
                start_date=start_date,
                end_date=end_date,
                timeframe="H1",
                initial_balance=10000.0,
                risk_per_trade=risk_percent / 100
            )
            
            if not result:
                return {
                    "valid": False,
                    "reason": "Backtest failed to run",
                    "metrics": {}
                }
            
            # Check if the results meet our criteria
            valid = (
                result.win_rate >= min_win_rate and
                result.profit_factor >= min_profit_factor and
                result.max_drawdown_pct <= max_drawdown
            )
            
            # If not valid, determine the reason
            reason = ""
            if not valid:
                if result.win_rate < min_win_rate:
                    reason += f"Win rate {result.win_rate:.1f}% below threshold {min_win_rate:.1f}%. "
                if result.profit_factor < min_profit_factor:
                    reason += f"Profit factor {result.profit_factor:.2f} below threshold {min_profit_factor:.2f}. "
                if result.max_drawdown_pct > max_drawdown:
                    reason += f"Max drawdown {result.max_drawdown_pct:.2f}% exceeds threshold {max_drawdown:.2f}%. "
            else:
                reason = "Trade validated by backtesting"
            
            # Return the validation result with metrics
            return {
                "valid": valid,
                "reason": reason.strip(),
                "metrics": {
                    "win_rate": result.win_rate,
                    "profit_factor": result.profit_factor,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown_pct": result.max_drawdown_pct,
                    "total_return_pct": result.total_return_pct,
                    "total_trades": result.total_trades
                }
            }
        
        except Exception as e:
            logger.error(f"Error in backtest validation: {e}")
            return {
                "valid": False,
                "reason": f"Error validating trade: {str(e)}",
                "metrics": {}
            }
    
    def _create_strategy_for_backtest(self, strategy_type, direction, entry_price, stop_loss, take_profit):
        """Create appropriate strategy for backtesting based on the trade parameters"""
        try:
            # Try to import from your brain module - adjust this import to match your project structure
            from brain import LLMGeneratedStrategy
        except ImportError:
            # If that fails, create a simplified class definition
            class LLMGeneratedStrategy:
                def __init__(self, name, parameters, strategy_code=None):
                    self.name = name
                    self.parameters = parameters
                    self.strategy_code = strategy_code
                    
                def generate_signals(self, data):
                    df = data.copy()
                    df['signal'] = 0
                    
                    direction = self.parameters.get('direction', 'BUY')
                    entry_price = self.parameters.get('entry_price', 0)
                    
                    if direction == 'BUY':
                        df.loc[df['close'] < entry_price, 'signal'] = 1
                    else:
                        df.loc[df['close'] > entry_price, 'signal'] = -1
                        
                    return df
        
        # Define strategy code based on the strategy type
        if strategy_type.lower() == "trend_following":
            strategy_code = f"""
def generate_signals(self, data):
    if data.empty:
        return data
    
    df = data.copy()
    
    # Calculate EMAs
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Initialize signal column
    df['signal'] = 0
    
    # Get trade direction and entry price
    direction = self.parameters.get('direction', '{direction}')
    entry_price = self.parameters.get('entry_price', {entry_price})
    
    # Define a price region (±5% around entry price)
    price_region = (df['close'] >= entry_price * 0.95) & (df['close'] <= entry_price * 1.05)
    
    if direction == 'BUY':
        # Buy signals when price is in region and trend is up
        buy_condition = price_region & (df['ema_20'] > df['ema_50'])
        df.loc[buy_condition, 'signal'] = 1
    else:
        # Sell signals when price is in region and trend is down
        sell_condition = price_region & (df['ema_20'] < df['ema_50'])
        df.loc[sell_condition, 'signal'] = -1
    
    return df
"""
        elif strategy_type.lower() == "breakout":
            strategy_code = f"""
def generate_signals(self, data):
    if data.empty:
        return data
    
    df = data.copy()
    
    # Calculate Bollinger Bands
    window = 20
    df['middle_band'] = df['close'].rolling(window=window).mean()
    df['std'] = df['close'].rolling(window=window).std()
    df['upper_band'] = df['middle_band'] + (df['std'] * 2)
    df['lower_band'] = df['middle_band'] - (df['std'] * 2)
    
    # Initialize signal column
    df['signal'] = 0
    
    # Get trade direction and entry price
    direction = self.parameters.get('direction', '{direction}')
    entry_price = self.parameters.get('entry_price', {entry_price})
    
    # Define a price region (±5% around entry price)
    price_region = (df['close'] >= entry_price * 0.95) & (df['close'] <= entry_price * 1.05)
    
    if direction == 'BUY':
        # Buy when price is near entry and breaks above upper band
        buy_condition = price_region & (df['close'] > df['upper_band'])
        df.loc[buy_condition, 'signal'] = 1
    else:
        # Sell when price is near entry and breaks below lower band
        sell_condition = price_region & (df['close'] < df['lower_band'])
        df.loc[sell_condition, 'signal'] = -1
    
    return df
"""
        else:  # Default to mean_reversion
            strategy_code = f"""
def generate_signals(self, data):
    if data.empty:
        return data
    
    df = data.copy()
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Initialize signal column
    df['signal'] = 0
    
    # Get trade direction and entry price
    direction = self.parameters.get('direction', '{direction}')
    entry_price = self.parameters.get('entry_price', {entry_price})
    
    # Define a price region (±5% around entry price)
    price_region = (df['close'] >= entry_price * 0.95) & (df['close'] <= entry_price * 1.05)
    
    if direction == 'BUY':
        # Buy when RSI is oversold and price in region
        buy_condition = price_region & (df['rsi'] < 30)
        df.loc[buy_condition, 'signal'] = 1
    else:
        # Sell when RSI is overbought and price in region
        sell_condition = price_region & (df['rsi'] > 70)
        df.loc[sell_condition, 'signal'] = -1
    
    return df
"""
    
        # Create strategy with proper parameters
        strategy = LLMGeneratedStrategy(
            name=f"Backtest_{strategy_type}_{direction}",
            parameters={
                "direction": direction,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "strategy_type": strategy_type
            },
            strategy_code=strategy_code
        )
        
        return strategy
    
    def validate_trade(self, trade_details, price_data, validation_period=10):
        """Validate a trade through backtesting before execution"""
        try:
            # Extract strategy details
            strategy_type = trade_details.get("strategy", "trend_following")
            direction = trade_details.get("direction", "BUY").upper()
            entry_price = float(trade_details.get("entry_price", 0))
            stop_loss = float(trade_details.get("stop_loss", 0))
            risk_percent = float(trade_details.get("risk_percent", self.max_total_risk_percent)) if trade_details.get("risk_percent") else self.max_total_risk_percent
            
            # Basic validation checks
            if entry_price <= 0 or stop_loss <= 0:
                return {"valid": False, "confidence": 0.0, "reason": "Invalid prices"}
                
            if risk_percent <= 0 or risk_percent > 5:
                # Cap risk at our maximum
                risk_percent = min(risk_percent, self.max_total_risk_percent)
                
            # Calculate risk-reward ratio
            take_profit_levels = trade_details.get("take_profit", [])
            if not take_profit_levels:
                return {"valid": False, "confidence": 0.0, "reason": "No take profit levels provided"}
                
            # Use first take profit level for calculation
            first_tp = float(take_profit_levels[0]) if isinstance(take_profit_levels, list) else float(take_profit_levels)
            
            # Calculate risk and reward
            if direction == "BUY":
                risk = entry_price - stop_loss
                reward = first_tp - entry_price
            else:  # SELL
                risk = stop_loss - entry_price
                reward = entry_price - first_tp
            
            # Validate risk-reward ratio
            risk_reward_ratio = reward / risk if risk > 0 else 0
            if risk_reward_ratio < 1.5:
                return {"valid": False, "confidence": 0.0, "reason": f"Poor risk-reward ratio: {risk_reward_ratio:.2f}"}
            
            # Basic confidence calculation
            confidence = min(0.8, 0.5 + (risk_reward_ratio - 1.5) * 0.1)
            
            # Check if there's a probability assessment we can use
            probability_data = trade_details.get("probability_assessment", {})
            if probability_data:
                # If the model provided confidence/win probability, use that
                win_prob = probability_data.get("win_probability", 0)
                if isinstance(win_prob, str):
                    # Convert string percentage to float
                    win_prob = float(win_prob.replace('%', '')) / 100
                
                if win_prob > 0:
                    # Blend our calculation with the model's assessment
                    confidence = (confidence + win_prob) / 2
            
            return {
                "valid": True,
                "confidence": confidence,
                "reason": f"Trade validated with {confidence:.2f} confidence, risk-reward: {risk_reward_ratio:.2f}"
            }
            
        except Exception as e:
            logger.error(f"Error validating trade: {e}")
            return {"valid": False, "confidence": 0.0, "reason": f"Validation error: {str(e)}"}
            
    def review_and_evolve(self, account_status):
        """Periodically review performance and evolve strategies (simplified version)"""
        try:
            logger.info("Reviewing trading performance and evolving strategies...")
            
            # Get recent trades
            recent_trades = []
            if hasattr(self.memory, "get_recent_trades"):
                recent_trades = self.memory.get_recent_trades(10)
            
            # Calculate basic performance metrics
            win_count = sum(1 for trade in recent_trades if trade.get("is_win", False))
            total_count = len(recent_trades)
            win_rate = win_count / total_count if total_count > 0 else 0
            
            # Log the performance review
            logger.info(f"Recent performance: {win_rate:.1%} win rate ({win_count}/{total_count})")
            
            # Get strategy weights
            strategy_weights = {}
            if hasattr(self.memory, "memory") and "strategy_weights" in self.memory.memory:
                strategy_weights = self.memory.memory.get("strategy_weights", {})
            
            # Simplified review response
            return {
                "performance_analysis": f"Reviewed performance with {win_rate:.1%} win rate ({win_count}/{total_count})",
                "strategy_weights": strategy_weights
            }
            
        except Exception as e:
            logger.error(f"Error during system review: {e}")
            return {"error": f"Review error: {str(e)}"}