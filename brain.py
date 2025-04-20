#!/usr/bin/env python3
"""

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
        
        # Build the prompt
        user_prompt = f"""
## Current Trading Status
- Account Balance: {account_data.get('balance')} {account_data.get('currency')}
- Safety Level: {self.memory.memory['safety_level']:.4f} (risk per trade)
- Daily Profit: {self.memory.memory['daily_profit_pct']:.2f}% (Target: 5%)
- Win Rate: {win_rate:.1f}% ({self.memory.memory['win_count']}/{self.memory.memory['trade_count']} trades)
- Open EUR/USD Positions: {len([p for p in positions if p.get('instrument') == 'EUR_USD'])}

## EUR/USD Recent Price Data (H1 Timeframe)
{json.dumps(simplified_candles, indent=2)}

## Technical Indicators
{json.dumps(technical_indicators, indent=2)}

## Multi-Timeframe Analysis
{json.dumps(multi_timeframe_data, indent=2)}

## Intermarket Correlations
{json.dumps(intermarket_data, indent=2)}

## Economic Calendar
{json.dumps(economic_data, indent=2)}

## Market Sentiment
{json.dumps(sentiment_data, indent=2)}

## Recent Trades
{json.dumps(recent_trades[:3], indent=2)}

## Current Strategy
{json.dumps(strategy_info, indent=2)}

## Strategy Weights
{json.dumps(self.memory.memory['strategy_weights'], indent=2)}

Based on this comprehensive market analysis:
1. Analyze EUR/USD across all timeframes and consider correlations with other markets
2. Consider economic events, sentiment data, and technical indicators
3. Decide if we should OPEN a new position, UPDATE an existing position, CREATE a new strategy, or WAIT
4. If opening, provide complete trade details with entry, stop, targets and risk
5. If updating, provide updated stop levels for proper risk management
6. If creating a new strategy, provide the strategy code and parameters
7. Consider multiple take-profit levels for staged profit-taking
8. Explain your reasoning based on multi-timeframe analysis and market conditions

If you believe we should create a new strategy, provide the code in Python format that would be used for backtesting before actual trading.

Respond in JSON format with: 
{
  "market_analysis": "Your comprehensive analysis including multiple timeframes and factors",
  "action": "OPEN, UPDATE, CREATE or WAIT",
  "trade_details": {
    "direction": "BUY or SELL",
    "entry_price": 1.xxxx,
    "entry_range": [min, max],
    "stop_loss": 1.xxxx,
    "take_profit": [level1, level2, level3],
    "risk_percent": "between 1-5",
    "trailing_stop_distance": 0.xxxx,
    "strategy": "trend_following, breakout, or mean_reversion",
    "reasoning": "Technical justification with multi-timeframe context"
  },
  "update_details": {
    "new_stop_loss": 1.xxxx,
    "reason": "Why update the stop"
  },
  "strategy_details": {
    "name": "Name of the new strategy",
    "description": "Detailed description of how the strategy works",
    "parameters": {
      "param1": value1,
      "param2": value2
    },
    "code": "Python code for the strategy generate_signals function",
    "backtest_period": "e.g., 30, 60, or 90 days"
  },
  "exit_strategy": {
    "early_exit_conditions": "Conditions to exit early if trade not performing as expected",
    "partial_profit_taking": "Detailed plan for taking profits at different levels"
  },
  "self_improvement": {
    "performance_assessment": "How your recent trades performed across different market conditions",
    "strategy_adjustments": "How you would adjust your strategy weights based on all available data",
    "prompt_improvements": "Suggestions to improve your system prompt"
  }
}"""
        
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
                         validation_period=30, confidence_threshold=0.6):
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
            
            # Import MIN_CONFIDENCE_THRESHOLD to avoid circular import
            from bot import MIN_CONFIDENCE_THRESHOLD
            
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
        
        # Build the review prompt 
        review_prompt = (
            "## Performance Review\n"
            f"- Account Balance: {account_data.get('balance')} {account_data.get('currency')}\n"
            f"- Daily P&L: {daily_profit_pct:.2f}% (Target: 5%)\n"
            f"- Win Rate: {win_rate:.1f}% ({self.memory.memory['win_count']}/{self.memory.memory['trade_count']} trades)\n"
            f"- Current Safety Level: {self.memory.memory['safety_level']:.4f}\n"
            f"- Strategy Weights: {json.dumps(self.memory.memory['strategy_weights'], indent=2)}\n\n"
            "## Current System Prompt\n"
            f"{self.system_prompt}\n\n"
            "## Recent Trades (Last 20)\n"
            f"{json.dumps(recent_trades, indent=2)}\n\n"
            "## Saved Strategies\n"
            f"{json.dumps([{'name': s['name'], 'parameters': s['parameters']} for s in saved_strategies], indent=2)}\n\n"
            "Based on this comprehensive review of your trading performance:\n\n"
            "1. Analyze what's working and what's not working in your trading approach\n"
            "2. Evaluate which strategies have been most effective\n"
            "3. Suggest modifications to your strategy weights\n"
            "4. Recommend improvements to your system prompt for better results\n"
            "5. Consider changes to your risk management approach\n\n"
            "Provide your evolution recommendations in JSON format:\n"
            "{\n"
            '  "performance_analysis": "Detailed analysis of trading performance",\n'
            '  "strategy_weights": {\n'
            '    "trend_following": "float", \n'
            '    "breakout": "float", \n'
            '    "mean_reversion": "float"\n'
            '  },\n'
            '  "risk_adjustments": "Recommendations for risk management",\n'
            '  "improved_prompt": "Complete improved system prompt",\n'
            '  "reasoning": "Detailed reasoning for all changes"\n'
            "}"
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

Trading Bot Brain Module
Contains the LLM integration and decision-making logic for the trading bot
"""

import os
import json
import logging
import datetime
import pandas as pd
import numpy as np
import openai
from typing import Dict, List, Any, Optional, Tuple, Union

# Configure logging
logger = logging.getLogger("Trading_Brain")

# --- LLM Generated Strategy ---
class LLMGeneratedStrategy:
    """Strategy dynamically generated by LLM with parameters"""
    
    def __init__(self, name="LLM Generated Strategy", parameters=None, strategy_code=None):
        """Initialize with strategy parameters and executable code"""
        super().__init__()
        self.name = name
        
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
    
    def calculate_stop_loss(self, data, signal_row_idx, direction):
        """Calculate stop loss based on recent price action"""
        # Look back for recent swing points
        lookback = min(20, signal_row_idx)
        recent_data = data.iloc[signal_row_idx-lookback:signal_row_idx+1]
        
        if direction == "BUY":
            # For buy signals, set stop below recent low
            stop_price = recent_data['low'].min() * 0.998  # Slightly below the low
        else:
            # For sell signals, set stop above recent high
            stop_price = recent_data['high'].max() * 1.002  # Slightly above the high
            
        return stop_price
    
    def calculate_take_profit(self, data, signal_row_idx, direction):
        """Calculate take profit based on stop loss distance"""
        # Set take profit at 2:1 risk-reward ratio
        entry_price = data.iloc[signal_row_idx]['close']
        stop_loss = self.calculate_stop_loss(data, signal_row_idx, direction)
        
        if direction == "BUY":
            risk = entry_price - stop_loss
            take_profit = entry_price + (risk * 2)
        else:
            risk = stop_loss - entry_price
            take_profit = entry_price - (risk * 2)
            
        return take_profit
    
    def set_parameters(self, **kwargs):
        """Update strategy parameters"""
        self.parameters.update(kwargs)
    
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
        from bot import INITIAL_SAFETY_LEVEL
        
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
        from bot import INCREASE_FACTOR
        
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

# --- LLM Trading Brain ---
class LLMTradingBrain:
    """LLM-powered trading decision maker with self-improvement capabilities"""
    
    def __init__(self, memory, backtester):
        """Initialize the LLM trading brain"""
        self.memory = memory
        self.backtester = backtester
        
        # Get API key from environment
        from bot import OPENAI_API_KEY
        openai.api_key = OPENAI_API_KEY
        
        # Initial system prompt (will be evolved over time)
        self.system_prompt = """
You are a self-evolving forex trading AI specializing in EUR/USD.

Your goal is to achieve a 5% daily return while respecting strict risk management:
- Maximum account risk: 10%
- Maximum daily drawdown: 10%
- Implement trailing stops and profit taking

Your decisions should be based on:
1. Technical analysis of EUR/USD price action
2. Analysis of past trades (what worked and what didn't)
3. Backtesting validation of strategies before trading
4. Risk management based on account's current exposure

Always provide a clear trade plan with:
- Direction (BUY/SELL)
- Entry price and acceptable range
- Stop loss level (must be specified)
- Take profit targets (multiple levels recommended)
- Risk percentage (1-5% based on conviction)
- Position sizing logic
- Technical justification

Consider using these strategies based on market conditions:
- Trend following: Identify and follow established trends
- Breakout: Capture price moves from established ranges
- Mean reversion: Trade returns to mean after extreme movements

Your memory and self-improvement:
- Learn from successful and unsuccessful trades
- Adapt your approach based on recent performance
- Evolve your strategies over time for better results
- Create and backtest new strategies before deployment
"""