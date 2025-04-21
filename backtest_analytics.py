#!/usr/bin/env python3
"""
Backtesting Analytics Module
Adds result categorization, statistics tracking, and dashboard functionality to the backtesting engine
"""

import os
import json
import time
import logging
import datetime
import threading
from typing import Dict, List, Any, Optional, Union

# Configure logging
logger = logging.getLogger("Backtest_Analytics")

class BacktestAnalytics:
    """
    Analytics and organization for backtesting results
    Designed to work alongside the existing backtesting engine
    """
    
    def __init__(self, base_dir: str = "backtest_results"):
        """Initialize with base directory for results"""
        self.base_dir = base_dir
        self.profitable_dir = os.path.join(base_dir, "profitable")
        self.losing_dir = os.path.join(base_dir, "losing")
        
        # Create directories if they don't exist
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.profitable_dir, exist_ok=True)
        os.makedirs(self.losing_dir, exist_ok=True)
        
        # Start dashboard update thread
        self.schedule_dashboard_updates()
    
    def save_categorized_backtest(self, result, trade_details=None) -> str:
        """
        Save backtest results in categorized folders based on profitability
        
        Args:
            result: BacktestResult object with performance data
            trade_details: Optional dict with additional trade information
            
        Returns:
            Path to saved file
        """
        # Determine if the result was profitable
        is_profitable = result.total_return > 0
        
        # Create a descriptive filename with performance metrics
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_name = result.strategy_name.replace(" ", "_").lower()
        instrument = result.instrument.replace("/", "").lower()
        return_pct = result.total_return_pct
        
        # Format the filename to be sortable by return percentage
        if is_profitable:
            # For profitable trades, format with return percentage for easy sorting
            filename = f"{self.profitable_dir}/return_{return_pct:.2f}_pct_{strategy_name}_{instrument}_{timestamp}.json"
        else:
            # For losing trades, format with negative return for sorting
            filename = f"{self.losing_dir}/return_minus{abs(return_pct):.2f}_pct_{strategy_name}_{instrument}_{timestamp}.json"
        
        # Add trade details if provided
        result_dict = result.to_dict()
        if trade_details:
            result_dict["trade_details"] = trade_details
        
        # Save the result
        with open(filename, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Backtest result saved to {filename}")
        
        # Update statistics after saving
        self.track_backtest_statistics()
        
        return filename
    
    def track_backtest_statistics(self) -> Dict[str, Any]:
        """
        Track overall statistics of backtests run
        
        Returns:
            Dict with statistics
        """
        stats_file = os.path.join(self.base_dir, "statistics.json")
        
        # Initialize or load current stats
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
        else:
            stats = {
                "total_backtests": 0,
                "profitable_backtests": 0,
                "losing_backtests": 0,
                "average_return": 0,
                "best_return": 0,
                "worst_return": 0,
                "best_strategy": "",
                "worst_strategy": ""
            }
        
        # Update statistics by counting files
        profitable_count = len(os.listdir(self.profitable_dir)) if os.path.exists(self.profitable_dir) else 0
        losing_count = len(os.listdir(self.losing_dir)) if os.path.exists(self.losing_dir) else 0
        
        stats["total_backtests"] = profitable_count + losing_count
        stats["profitable_backtests"] = profitable_count
        stats["losing_backtests"] = losing_count
        
        # Calculate best and worst returns if files exist
        if profitable_count > 0:
            best_file = self._get_top_file(self.profitable_dir)
            if best_file:
                try:
                    with open(best_file, 'r') as f:
                        best_data = json.load(f)
                    stats["best_return"] = best_data.get("total_return_pct", 0)
                    stats["best_strategy"] = best_data.get("strategy_name", "unknown")
                except:
                    pass
        
        if losing_count > 0:
            worst_file = self._get_top_file(self.losing_dir)
            if worst_file:
                try:
                    with open(worst_file, 'r') as f:
                        worst_data = json.load(f)
                    stats["worst_return"] = worst_data.get("total_return_pct", 0)
                    stats["worst_strategy"] = worst_data.get("strategy_name", "unknown")
                except:
                    pass
        
        # Calculate average return across all backtests
        total_return = 0
        total_count = 0
        
        for directory in [self.profitable_dir, self.losing_dir]:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    try:
                        with open(os.path.join(directory, file), 'r') as f:
                            data = json.load(f)
                        total_return += data.get("total_return_pct", 0)
                        total_count += 1
                    except:
                        continue
        
        if total_count > 0:
            stats["average_return"] = total_return / total_count
        
        # Save updated stats
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
            
        return stats
    
    def _get_top_file(self, directory: str) -> Optional[str]:
        """Helper to get the top file in a directory based on sorting"""
        if not os.path.exists(directory) or not os.listdir(directory):
            return None
            
        files = os.listdir(directory)
        files.sort(reverse=True)  # Relies on filename sorting
        
        if files:
            return os.path.join(directory, files[0])
        return None
    
    def get_best_strategies(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Find the best-performing strategies from backtest results
        
        Args:
            top_n: Number of top strategies to return
            
        Returns:
            List of dicts with strategy information
        """
        if not os.path.exists(self.profitable_dir):
            return []
            
        files = os.listdir(self.profitable_dir)
        
        # Extract return percentages from filenames
        strategies = []
        for file in files:
            try:
                # Parse filename like "return_2.45_pct_strategy_name_instrument_timestamp.json"
                parts = file.split('_')
                return_pct = float(parts[1])
                strategy_name = '_'.join(parts[3:-2])  # Extract strategy name
                
                strategies.append({
                    "strategy": strategy_name,
                    "return_pct": return_pct,
                    "file": os.path.join(self.profitable_dir, file)
                })
            except:
                continue
        
        # Sort by return percentage (descending)
        strategies.sort(key=lambda x: x["return_pct"], reverse=True)
        
        return strategies[:top_n]
    
    def calculate_strategy_success_rates(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate success rates for different strategies
        
        Returns:
            Dict mapping strategy names to success statistics
        """
        # Get all backtest files
        all_files = []
        if os.path.exists(self.profitable_dir):
            all_files.extend([os.path.join(self.profitable_dir, f) for f in os.listdir(self.profitable_dir)])
        if os.path.exists(self.losing_dir):
            all_files.extend([os.path.join(self.losing_dir, f) for f in os.listdir(self.losing_dir)])
        
        # Count successes/failures by strategy
        strategy_stats = {}
        
        for file_path in all_files:
            try:
                with open(file_path, 'r') as f:
                    result = json.load(f)
                
                strategy_name = result.get("strategy_name", "unknown")
                is_profitable = "profitable" in file_path
                
                if strategy_name not in strategy_stats:
                    strategy_stats[strategy_name] = {"wins": 0, "losses": 0}
                
                if is_profitable:
                    strategy_stats[strategy_name]["wins"] += 1
                else:
                    strategy_stats[strategy_name]["losses"] += 1
                    
            except:
                continue
        
        # Calculate success rates
        for strategy, stats in strategy_stats.items():
            total = stats["wins"] + stats["losses"]
            if total > 0:
                stats["success_rate"] = (stats["wins"] / total) * 100
            else:
                stats["success_rate"] = 0
        
        return strategy_stats
    
    def analyze_backtest_conditions(self, top_n: int = 10) -> Dict[str, Any]:
        """
        Analyze market conditions in successful vs. unsuccessful backtests
        
        Args:
            top_n: Number of top indicators to return
            
        Returns:
            Dict with analysis results
        """
        # Collect market conditions data
        profitable_conditions = []
        losing_conditions = []
        
        # Process profitable backtests
        if os.path.exists(self.profitable_dir):
            for file in os.listdir(self.profitable_dir)[:50]:  # Limit to recent files
                try:
                    with open(os.path.join(self.profitable_dir, file), 'r') as f:
                        data = json.load(f)
                        
                    if "trade_details" in data and "market_analysis" in data.get("trade_details", {}):
                        conditions = {
                            "return_pct": data.get("total_return_pct", 0),
                            "market_conditions": data.get("trade_details", {}).get("market_analysis", ""),
                            "direction": data.get("trade_details", {}).get("direction", "UNKNOWN")
                        }
                        profitable_conditions.append(conditions)
                except:
                    continue
        
        # Process losing backtests
        if os.path.exists(self.losing_dir):
            for file in os.listdir(self.losing_dir)[:50]:  # Limit to recent files
                try:
                    with open(os.path.join(self.losing_dir, file), 'r') as f:
                        data = json.load(f)
                        
                    if "trade_details" in data and "market_analysis" in data.get("trade_details", {}):
                        conditions = {
                            "return_pct": data.get("total_return_pct", 0),
                            "market_conditions": data.get("trade_details", {}).get("market_analysis", ""),
                            "direction": data.get("trade_details", {}).get("direction", "UNKNOWN")
                        }
                        losing_conditions.append(conditions)
                except:
                    continue
        
        # Find common phrases in profitable conditions
        all_profitable_text = " ".join([c["market_conditions"] for c in profitable_conditions])
        all_losing_text = " ".join([c["market_conditions"] for c in losing_conditions])
        
        # Simple analysis of word frequency
        profitable_words = [word.lower() for word in all_profitable_text.split() 
                           if len(word) > 4 and word.isalpha()]
        losing_words = [word.lower() for word in all_losing_text.split() 
                       if len(word) > 4 and word.isalpha()]
        
        # Count word frequencies
        profitable_freq = {}
        for word in profitable_words:
            profitable_freq[word] = profitable_freq.get(word, 0) + 1
        
        losing_freq = {}
        for word in losing_words:
            losing_freq[word] = losing_freq.get(word, 0) + 1
        
        # Find distinctive words for profitable trades
        distinctive_profitable = {}
        for word, count in profitable_freq.items():
            if count >= 3:  # Word appears at least 3 times
                losing_count = losing_freq.get(word, 0)
                if losing_count == 0:
                    # Word appears in profitable but not losing
                    distinctive_profitable[word] = count
                elif count / len(profitable_conditions) > (losing_count / len(losing_conditions)) * 2 if len(losing_conditions) > 0 else True:
                    # Word appears proportionally twice as often in profitable
                    distinctive_profitable[word] = count
        
        # Sort by frequency
        sorted_distinctive = sorted(distinctive_profitable.items(), 
                                  key=lambda x: x[1], reverse=True)
        
        # Analyze trade directions
        buy_profitable = sum(1 for c in profitable_conditions if c["direction"] == "BUY")
        sell_profitable = sum(1 for c in profitable_conditions if c["direction"] == "SELL")
        buy_losing = sum(1 for c in losing_conditions if c["direction"] == "BUY")
        sell_losing = sum(1 for c in losing_conditions if c["direction"] == "SELL")
        
        total_profitable = len(profitable_conditions)
        total_losing = len(losing_conditions)
        
        buy_success_rate = (buy_profitable / (buy_profitable + buy_losing)) * 100 if (buy_profitable + buy_losing) > 0 else 0
        sell_success_rate = (sell_profitable / (sell_profitable + sell_losing)) * 100 if (sell_profitable + sell_losing) > 0 else 0
        
        return {
            "top_profitable_indicators": sorted_distinctive[:top_n],
            "direction_analysis": {
                "buy_success_rate": buy_success_rate,
                "sell_success_rate": sell_success_rate,
                "buy_count": buy_profitable + buy_losing,
                "sell_count": sell_profitable + sell_losing
            },
            "sample_size": {
                "profitable": total_profitable,
                "losing": total_losing
            }
        }
    
    def generate_performance_dashboard(self) -> str:
        """
        Generate a simple HTML dashboard of backtest performance
        
        Returns:
            Path to the generated dashboard HTML file
        """
        dashboard_file = os.path.join(self.base_dir, "dashboard.html")
        
        # Get backtest statistics
        stats = self.track_backtest_statistics()
        best_strategies = self.get_best_strategies(top_n=5)
        strategy_success_rates = self.calculate_strategy_success_rates()
        
        # Get market condition analysis if we have enough data
        total_backtests = stats.get("total_backtests", 0)
        if total_backtests >= 10:
            market_conditions = self.analyze_backtest_conditions()
        else:
            market_conditions = {
                "top_profitable_indicators": [],
                "direction_analysis": {
                    "buy_success_rate": 0,
                    "sell_success_rate": 0,
                    "buy_count": 0,
                    "sell_count": 0
                },
                "sample_size": {
                    "profitable": 0,
                    "losing": 0
                }
            }
        
        # Create dashboard HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Trading Bot Backtest Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .card {{ border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 20px; }}
        .card h2 {{ margin-top: 0; color: #333; }}
        .good {{ color: green; }}
        .bad {{ color: red; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Backtest Performance Dashboard</h1>
    <p>Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="card">
        <h2>Overall Statistics</h2>
        <table>
            <tr><td>Total Backtests</td><td>{stats.get('total_backtests', 0)}</td></tr>
            <tr><td>Profitable Backtests</td><td class="good">{stats.get('profitable_backtests', 0)}</td></tr>
            <tr><td>Losing Backtests</td><td class="bad">{stats.get('losing_backtests', 0)}</td></tr>
            <tr><td>Success Rate</td><td>{stats.get('profitable_backtests', 0) / max(1, stats.get('total_backtests', 1)) * 100:.2f}%</td></tr>
            <tr><td>Average Return</td><td>{stats.get('average_return', 0):.2f}%</td></tr>
            <tr><td>Best Return</td><td class="good">{stats.get('best_return', 0):.2f}%</td></tr>
            <tr><td>Worst Return</td><td class="bad">{stats.get('worst_return', 0):.2f}%</td></tr>
        </table>
    </div>
    
    <div class="card">
        <h2>Top Performing Strategies</h2>
        <table>
            <tr><th>Strategy</th><th>Return %</th></tr>
"""
        
        # Add best strategies
        for strategy in best_strategies:
            html += f"""<tr><td>{strategy['strategy']}</td><td class="good">{strategy['return_pct']:.2f}%</td></tr>\n"""
        
        if not best_strategies:
            html += """<tr><td colspan="2">No profitable strategies recorded yet</td></tr>\n"""
        
        html += """
        </table>
    </div>
    
    <div class="card">
        <h2>Strategy Success Rates</h2>
        <table>
            <tr><th>Strategy</th><th>Success Rate</th><th>Win/Loss</th></tr>
"""
        
        # Add strategy success rates
        if strategy_success_rates:
            for strategy, data in strategy_success_rates.items():
                html += f"""<tr><td>{strategy}</td><td>{data.get('success_rate', 0):.2f}%</td><td>{data.get('wins', 0)}/{data.get('losses', 0)}</td></tr>\n"""
        else:
            html += """<tr><td colspan="3">No strategy data recorded yet</td></tr>\n"""
        
        html += """
        </table>
    </div>
    
    <div class="card">
        <h2>Market Indicators in Profitable Trades</h2>
        <table>
            <tr><th>Indicator</th><th>Frequency</th></tr>
"""
        
        # Add market indicators
        if market_conditions.get('top_profitable_indicators'):
            for word, count in market_conditions.get('top_profitable_indicators', []):
                html += f"""<tr><td>{word}</td><td>{count}</td></tr>\n"""
        else:
            html += """<tr><td colspan="2">Not enough data to identify indicators yet</td></tr>\n"""
        
        html += """
        </table>
    </div>
    
    <div class="card">
        <h2>Trade Direction Analysis</h2>
        <table>
            <tr><th>Direction</th><th>Success Rate</th><th>Count</th></tr>
"""
        
        # Add direction analysis
        direction_data = market_conditions.get('direction_analysis', {})
        if direction_data.get('buy_count', 0) > 0 or direction_data.get('sell_count', 0) > 0:
            html += f"""<tr><td>BUY</td><td>{direction_data.get('buy_success_rate', 0):.2f}%</td><td>{direction_data.get('buy_count', 0)}</td></tr>\n"""
            html += f"""<tr><td>SELL</td><td>{direction_data.get('sell_success_rate', 0):.2f}%</td><td>{direction_data.get('sell_count', 0)}</td></tr>\n"""
        else:
            html += """<tr><td colspan="3">Not enough direction data recorded yet</td></tr>\n"""
        
        html += """
        </table>
    </div>
</body>
</html>"""
        
        # Write to file
        with open(dashboard_file, 'w') as f:
            f.write(html)
        
        return dashboard_file
    
    def schedule_dashboard_updates(self, interval: int = 3600) -> None:
        """
        Schedule periodic dashboard updates
        
        Args:
            interval: Update interval in seconds (default: 1 hour)
        """
        def update_dashboard():
            while True:
                try:
                    self.generate_performance_dashboard()
                    logger.info("Dashboard updated")
                except Exception as e:
                    logger.error(f"Error updating dashboard: {e}")
                
                # Update on schedule
                time.sleep(interval)
        
        # Start dashboard update thread
        dashboard_thread = threading.Thread(target=update_dashboard, daemon=True)
        dashboard_thread.start()
        logger.info(f"Dashboard update scheduler started (interval: {interval}s)")

# Helper function to enhance existing Backtester class
def enhance_backtester(backtester_class):
    """
    Factory function to enhance an existing Backtester class with analytics
    
    Args:
        backtester_class: The original Backtester class
        
    Returns:
        Enhanced Backtester class with analytics capabilities
    """
    original_init = backtester_class.__init__
    original_run_backtest = backtester_class.run_backtest
    
    def enhanced_init(self, *args, **kwargs):
        # Call original init
        original_init(self, *args, **kwargs)
        # Add analytics
        self.analytics = BacktestAnalytics()
    
    def enhanced_run_backtest(self, *args, **kwargs):
        # Call original backtest method
        result = original_run_backtest(self, *args, **kwargs)
        
        # Save result if valid
        if result and hasattr(self, 'analytics'):
            # Extract trade details if present in kwargs
            trade_details = kwargs.get('trade_details', None)
            self.analytics.save_categorized_backtest(result, trade_details)
        
        return result
    
    # Create new class
    class EnhancedBacktester(backtester_class):
        pass
    
    # Apply enhancements
    EnhancedBacktester.__init__ = enhanced_init
    EnhancedBacktester.run_backtest = enhanced_run_backtest
    
    # Add direct analytics methods
    EnhancedBacktester.get_best_strategies = lambda self: self.analytics.get_best_strategies()
    EnhancedBacktester.calculate_strategy_success_rates = lambda self: self.analytics.calculate_strategy_success_rates()
    EnhancedBacktester.analyze_backtest_conditions = lambda self: self.analytics.analyze_backtest_conditions()
    EnhancedBacktester.generate_performance_dashboard = lambda self: self.analytics.generate_performance_dashboard()
    
    return EnhancedBacktester

# Example usage
if __name__ == "__main__":
    print("Backtesting Analytics module - Run with an existing backtester instance")