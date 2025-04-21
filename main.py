#!/usr/bin/env python3
"""
Main entry point for the LLM-powered Forex Trading Bot
Using standard LLMTradingBrain rather than EnhancedTradingBrain
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
# We'll use the standard brain instead of the enhanced version
from brain import LLMTradingBrain

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
        logger.info("Initializing EUR/USD Trading Bot")
        logger.info("Risk profile: max 2% total account exposure, max 1% risk per trade")
        
        # Create trading bot
        trading_bot = EURUSDTradingBot()
        
        # Make sure the bot has the error tracking attribute
        if not hasattr(trading_bot, 'last_error'):
            trading_bot.last_error = None
        
        # Note: We're not modifying the brain as it's already set in EURUSDTradingBot.__init__
        # with both the memory and backtester parameters
        
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
                
                # Check if daily target or max drawdown reached
                if account_status.get("target_reached", False):
                    logger.info(f"Daily target of {account_status['daily_profit_pct']:.2f}% reached. Pausing trading.")
                    time.sleep(600)  # Wait 10 minutes before checking again
                    continue
                    
                if account_status.get("max_drawdown_reached", False):
                    logger.warning(f"Maximum daily drawdown of {account_status['daily_drawdown']:.2f}% reached. Pausing trading.")
                    time.sleep(600)  # Wait 10 minutes before checking again
                    continue
                
                # Get current positions
                positions = trading_bot.oanda_client.get_open_positions()
                
                # Calculate current exposure as percentage of account balance
                current_exposure = 0.0
                if positions:
                    eur_usd_positions = [p for p in positions if p.get('instrument') == 'EUR_USD']
                    for position in eur_usd_positions:
                        # Get position units
                        long_units = int(position.get('long', {}).get('units', 0))
                        short_units = int(position.get('short', {}).get('units', 0))
                        
                        # Use the absolute value of whichever is non-zero
                        units = abs(long_units) if long_units != 0 else abs(short_units)
                        
                        # Get the average price
                        avg_price = float(position.get('long', {}).get('averagePrice', 0) or position.get('short', {}).get('averagePrice', 0))
                        
                        # Calculate position value (simplified)
                        position_value = units * avg_price * 0.0001  # Simplified value calculation
                        
                        # Calculate exposure percentage
                        exposure = (position_value / account_status['balance']) * 100
                        
                        current_exposure += exposure
                        logger.info(f"Position units: {units}, Value: {position_value}, Exposure: {exposure:.2f}%")
                
                logger.info(f"Current exposure: {current_exposure:.2f}% of account balance")
                
                # Get price data for analysis
                price_data = trading_bot.oanda_client.get_eur_usd_data(count=100, granularity="H1")
                
                # Get additional market data if available
                market_data = {}
                try:
                    # Add multiple timeframes if available
                    if hasattr(trading_bot.oanda_client, 'get_multi_timeframe_data'):
                        market_data["multi_timeframe"] = trading_bot.oanda_client.get_multi_timeframe_data()
                except Exception as e:
                    logger.warning(f"Error collecting additional market data: {e}")
                
                # If we have no price data, wait and try again
                if price_data.empty:
                    logger.warning("No price data received. Waiting...")
                    time.sleep(60)  # Wait 1 minute
                    continue
                
                # Get trading decision
                logger.info("Requesting market analysis...")
                decision = trading_bot.brain.analyze_market(
                    price_data=price_data,
                    account_data=account_status,
                    positions=positions,
                    market_data=market_data
                )
                
                # Log decision
                logger.info(f"Trading decision: {decision.get('action', 'UNKNOWN')}")
                
                # Execute decision with basic validation
                if decision.get("action", "WAIT") != "WAIT":
                    # Validate decision based on current position state
                    has_positions = len([p for p in positions if p.get('instrument') == 'EUR_USD']) > 0
                    
                    # Logical validation for UPDATE and CLOSE actions
                    if decision.get("action") in ["UPDATE", "CLOSE"] and not has_positions:
                        logger.warning(f"Cannot {decision.get('action')} when no position exists. Skipping execution.")
                        trading_bot.last_error = f"Cannot {decision.get('action')} when no position exists"
                    # Logical validation for OPEN action
                    elif decision.get("action") == "OPEN" and has_positions:
                        logger.warning("Cannot OPEN when position already exists. Skipping execution.")
                        trading_bot.last_error = "Cannot OPEN when position already exists"
                    else:
                        # Proceed with execution if validation passes
                        result = False
                        
                        # Execute the decision
                        if hasattr(trading_bot, 'execute_decision'):
                            result = trading_bot.execute_decision(decision, price_data)
                        else:
                            logger.error("Trading bot doesn't have execute_decision method")
                            trading_bot.last_error = "Missing execute_decision method in trading bot"
                        
                        if result:
                            logger.info(f"Successfully executed {decision.get('action')} action")
                        else:
                            # Get more detailed error information
                            error_reason = getattr(trading_bot, 'last_error', 'Unknown reason')
                            logger.warning(f"Failed to execute {decision.get('action')} action: {error_reason}")
                else:
                    logger.info("Decision is WAIT, no action taken")
                
                # Determine appropriate wait time
                if account_status.get("target_reached", False) or account_status.get("max_drawdown_reached", False):
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
                logger.error(f"Error in main loop: {e}", exc_info=True)
                # Check if it's a connection error and sleep longer
                if "Connection" in str(e) or "10054" in str(e):
                    logger.info("Connection error detected. Waiting 5 minutes before retrying...")
                    time.sleep(300)  # 5 minutes
                else:
                    logger.info("Waiting 60 seconds before retrying...")
                    time.sleep(60)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())