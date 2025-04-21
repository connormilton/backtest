#!/usr/bin/env python3
"""
Trading Memory Module
Maintains a history of trades and system state for the trading bot
"""

import os
import json
import logging
import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logger = logging.getLogger("Trading_Memory")

class TradingMemory:
    """Maintains trading history and system state"""
    
    def __init__(self):
        """Initialize trading memory"""
        self.memory_file = "data/system_memory.json"
        self.trade_log_file = "data/trade_log.jsonl"
        self.review_log_file = "data/review_log.jsonl"
        self.day_start_file = "data/day_start_info.json"
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        # Initialize memory
        self.memory = {
            "created": datetime.datetime.now().isoformat(),
            "last_updated": datetime.datetime.now().isoformat(),
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "safety_level": 0.01,  # 1% default safety level
            "strategy_weights": {
                "trend_following": 1.0,
                "breakout": 1.0,
                "mean_reversion": 1.0
            }
        }
        
        # Load memory if it exists
        self.load_memory()
    
    def load_memory(self):
        """Load system memory from file if it exists"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r") as f:
                    self.memory = json.load(f)
                logger.info("Loaded system memory from file")
            except Exception as e:
                logger.error(f"Error loading memory: {e}")
                # Initialize with default values (already done in __init__)
    
    def save_memory(self):
        """Save system memory to file"""
        try:
            self.memory["last_updated"] = datetime.datetime.now().isoformat()
            with open(self.memory_file, "w") as f:
                json.dump(self.memory, f, indent=2)
            logger.info("Saved system memory to file")
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
    
    def record_trade(self, direction=None, entry_price=None, stop_loss=None, 
                    take_profit=None, risk_percent=None, exit_price=None, 
                    is_win=None, is_loss=None, reasoning=None, strategy=None):
        """Record a trade to the trade log"""
        try:
            # Create trade record
            trade = {
                "timestamp": datetime.datetime.now().isoformat(),
                "direction": direction,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_percent": risk_percent,
                "exit_price": exit_price,
                "is_win": is_win,
                "is_loss": is_loss,
                "reasoning": reasoning,
                "strategy": strategy
            }
            
            # Update trade statistics
            self.memory["trade_count"] += 1
            if is_win:
                self.memory["win_count"] += 1
            elif is_loss:
                self.memory["loss_count"] += 1
            
            # Save to trade log
            with open(self.trade_log_file, "a") as f:
                f.write(json.dumps(trade) + "\n")
                
            # Save memory updates
            self.save_memory()
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
            return False
    
    def update_last_trade(self, exit_price=None, is_win=None, is_loss=None, is_closed=False):
        """Update the most recent trade with exit information"""
        try:
            # Get the last trade
            recent_trades = self.get_recent_trades(1)
            if not recent_trades:
                logger.warning("No trades found to update")
                return False
            
            last_trade = recent_trades[0]
            
            # Update trade data
            if exit_price is not None:
                last_trade["exit_price"] = exit_price
            
            if is_win is not None:
                last_trade["is_win"] = is_win
                if is_win:
                    self.memory["win_count"] += 1
            
            if is_loss is not None:
                last_trade["is_loss"] = is_loss
                if is_loss:
                    self.memory["loss_count"] += 1
            
            if is_closed:
                last_trade["closed_at"] = datetime.datetime.now().isoformat()
            
            # Replace the last line in the trade log
            with open(self.trade_log_file, "r") as f:
                lines = f.readlines()
            
            if lines:
                # Replace the last line with updated trade
                lines[-1] = json.dumps(last_trade) + "\n"
                
                with open(self.trade_log_file, "w") as f:
                    f.writelines(lines)
            
            # Save memory updates
            self.save_memory()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating last trade: {e}")
            return False
    
    def get_recent_trades(self, limit=5):
        """Get recent trades from the trade log"""
        try:
            trades = []
            
            if os.path.exists(self.trade_log_file):
                with open(self.trade_log_file, "r") as f:
                    for line in f:
                        try:
                            trade = json.loads(line.strip())
                            trades.append(trade)
                        except:
                            continue
                            
                # Return most recent trades first
                trades.reverse()
                
                return trades[:limit]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
            return []
    
    def get_day_start_account_info(self):
        """Get account info at the start of the day"""
        try:
            if os.path.exists(self.day_start_file):
                with open(self.day_start_file, "r") as f:
                    return json.load(f)
            
            # Return empty info if no file exists
            return {}
            
        except Exception as e:
            logger.error(f"Error getting day start info: {e}")
            return {}
    
    def set_day_start_account_info(self, info):
        """Set account info at the start of the day"""
        try:
            # Ensure date is stored as string
            if "date" in info and not isinstance(info["date"], str):
                info["date"] = info["date"].isoformat()
                
            with open(self.day_start_file, "w") as f:
                json.dump(info, f, indent=2)
                
            return True
            
        except Exception as e:
            logger.error(f"Error setting day start info: {e}")
            return False
    
    def get_saved_strategies(self, limit=3):
        """Get saved strategies from memory"""
        try:
            strategies = self.memory.get("saved_strategies", [])
            return strategies[:limit]
        except Exception as e:
            logger.error(f"Error getting saved strategies: {e}")
            return []

# Example usage
if __name__ == "__main__":
    print("Trading Memory Module - For use with the Trading Bot")