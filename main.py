#!/usr/bin/env python3
"""
Main entry point for the LLM-powered Forex Trading Bot
Updated to use Enhanced Trading Brain with backtesting integration
Includes updated risk parameters: max 2% total exposure, max 1% per trade
"""

import os
import sys
import time
import logging
import datetime
import pandas as pd
from dotenv import load_dotenv

# Import bot components
from bot import EURUSDTradingBot
# Import the enhanced brain
from enhanced_brain import EnhancedTradingBrain
from backtest_engine import Backtester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Main")

# Load environment variables
load_dotenv()

# Create required directories
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("backtest_results", exist_ok=True)
os.makedirs("strategies", exist_ok=True)

def main():
    """Main function to run the trading bot"""
    try:
        logger.info("Initializing EUR/USD Trading Bot with Enhanced Brain")
        logger.info("Risk profile: max 2% total account exposure, max 1% risk per trade")
        
        # Create trading bot
        trading_bot = EURUSDTradingBot()
        
        # Replace the brain with the enhanced version
        trading_bot.brain = EnhancedTradingBrain(trading_bot.memory, trading_bot.backtester)
        logger.info("Enhanced trading brain initialized with backtesting integration")
        
        # Main trading loop
        logger.info("Starting main trading loop")
        
        while True:
            try:
                # Get current time
                now = datetime.datetime.now()
                logger.info(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Check account status
                account_status = trading_bot.update_account_status()
                logger.info(f"Account status: Balance={account_status['balance']}, " +
                           f"Daily P&L={account_status['daily_profit_pct']:.2f}%, " +
                           f"Drawdown={account_status['daily_drawdown']:.2f}%")
                
                # Check current exposure
                current_exposure = trading_bot.calculate_current_exposure()
                logger.info(f"Current exposure: {current_exposure:.2f}% of account balance")
                
                # Check if daily target or max drawdown reached
                if account_status["target_reached"]:
                    logger.info(f"Daily target of {account_status['daily_profit_pct']:.2f}% reached. Pausing trading.")
                    time.sleep(600)  # Wait 10 minutes before checking again
                    continue
                    
                if account_status["max_drawdown_reached"]:
                    logger.warning(f"Maximum daily drawdown of {account_status['daily_drawdown']:.2f}% reached. Pausing trading.")
                    time.sleep(600)  # Wait 10 minutes before checking again
                    continue
                
                # Get current positions
                positions = trading_bot.oanda_client.get_open_positions()
                
                # Get H1 data for main analysis first
                price_data = trading_bot.oanda_client.get_eur_usd_data(count=100, granularity="H1")
                
                # Get additional market data if available
                market_data = {}
                try:
                    # Add multiple timeframes if available
                    market_data["multi_timeframe"] = trading_bot.oanda_client.get_multi_timeframe_data()
                    
                    # Add technical indicators if method exists and price data is available
                    if hasattr(trading_bot.oanda_client, 'get_technical_indicators') and not price_data.empty:
                        tech_indicators = trading_bot.oanda_client.get_technical_indicators(price_data)
                        market_data["technical_indicators"] = tech_indicators
                        
                    # Add intermarket data if method exists
                    if hasattr(trading_bot.oanda_client, 'get_intermarket_data'):
                        market_data["intermarket"] = trading_bot.oanda_client.get_intermarket_data()
                        
                    # Add economic data if method exists
                    if hasattr(trading_bot.oanda_client, 'get_economic_calendar'):
                        market_data["economic"] = trading_bot.oanda_client.get_economic_calendar()
                        
                    # Add sentiment data if method exists
                    if hasattr(trading_bot.oanda_client, 'get_market_sentiment'):
                        market_data["sentiment"] = trading_bot.oanda_client.get_market_sentiment()
                except Exception as e:
                    logger.warning(f"Error collecting additional market data: {e}")
                
                # If we have no price data, wait and try again
                if price_data.empty:
                    logger.warning("No price data received. Waiting...")
                    time.sleep(60)  # Wait 1 minute
                    continue
                
                # Get trading decision from enhanced brain
                logger.info("Requesting market analysis from enhanced brain...")
                decision = trading_bot.brain.analyze_market(
                    price_data=price_data,
                    account_data=account_status,
                    positions=positions,
                    market_data=market_data
                )
                
                # Log decision
                logger.info(f"Trading decision: {decision.get('action', 'UNKNOWN')}")
                
                # Check if backtesting was performed
                if "backtest_validation" in decision:
                    backtest_result = decision["backtest_validation"]
                    logger.info(f"Backtest validation: {backtest_result.get('valid')}, {backtest_result.get('reason')}")
                    
                    if backtest_result.get("metrics"):
                        metrics = backtest_result["metrics"]
                        logger.info(f"Backtest metrics: Win Rate: {metrics.get('win_rate', 0):.1f}%, " +
                                   f"Profit Factor: {metrics.get('profit_factor', 0):.2f}, " +
                                   f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
                
                # Check for risk compliance before executing decision
                if decision.get("action") == "OPEN":
                    trade_details = decision.get("trade_details", {})
                    risk_percent = trade_details.get("risk_percent", 1.0)
                    
                    # Ensure risk percent is a number and capped at 1%
                    if isinstance(risk_percent, str):
                        risk_percent = float(''.join(c for c in risk_percent if c.isdigit() or c == '.'))
                    risk_percent = min(1.0, float(risk_percent))
                    
                    # Update the risk parameter to ensure 1% max
                    trade_details["risk_percent"] = risk_percent
                    logger.info(f"Risk parameter capped at {risk_percent:.2f}%")
                    
                    # Check if new trade would exceed total exposure limit
                    if current_exposure + risk_percent > 2.0:
                        logger.warning(f"Trade rejected - would exceed 2% total risk limit: current {current_exposure:.2f}% + new {risk_percent:.2f}%")
                        decision["action"] = "WAIT"
                        decision["reason"] = f"Exceeded maximum total exposure of 2% (current: {current_exposure:.2f}%)"
                
                # Execute decision
                if decision.get("action") != "WAIT":
                    result = trading_bot.execute_decision(decision, price_data)
                    if result:
                        logger.info(f"Successfully executed {decision.get('action')} action")
                    else:
                        logger.warning(f"Failed to execute {decision.get('action')} action")
                
                # Review and evolve periodically (once per day)
                if now.hour == 0 and now.minute < 5:  # Around midnight
                    logger.info("Performing daily system review and evolution")
                    review_result = trading_bot.brain.review_and_evolve(account_status)
                    logger.info(f"System review completed: {review_result.get('performance_analysis', '')[:100]}...")
                
                # Determine appropriate wait time based on conditions
                if account_status["target_reached"] or account_status["max_drawdown_reached"]:
                    # Longer wait time when targets/limits reached
                    wait_time = 600  # 10 minutes
                elif now.hour < 7 or now.hour > 20:
                    # Longer interval outside main trading hours (assuming London hours)
                    wait_time = 900  # 15 minutes
                elif decision.get("action") != "WAIT" and decision.get("action") in ["OPEN", "UPDATE"]:
                    # Shorter interval after taking action
                    wait_time = 180  # 3 minutes
                else:
                    # Default wait time
                    wait_time = 300  # 5 minutes

                logger.info(f"Waiting {wait_time} seconds until next cycle...")
                time.sleep(wait_time)
                
            except KeyboardInterrupt:
                logger.info("Manual interruption detected. Exiting...")
                break
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                # Check if it's a connection error and sleep longer
                if "Connection" in str(e) or "10054" in str(e):
                    logger.info("Connection error detected. Waiting 5 minutes before retrying...")
                    time.sleep(300)  # 5 minutes
                else:
                    logger.info("Waiting 60 seconds before retrying...")
                    time.sleep(60)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())