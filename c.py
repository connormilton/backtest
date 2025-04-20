#!/usr/bin/env python3
"""
Simple EUR/USD Trading Bot using Polygon.io for data
Backtest forex strategies with detailed debugging
"""

import os
import json
import time
import requests
import logging
import argparse
import datetime
import pandas as pd
import numpy as np
import backtrader as bt
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv

# Create logger with detailed formatting
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler("polygon_bot_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Polygon_Bot")

# Create output directories
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Load environment variables
load_dotenv()

# Get API credentials from environment variables
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# Validate API credentials
if not POLYGON_API_KEY:
    logger.error("Polygon API key not found. Set POLYGON_API_KEY in .env file.")
    raise ValueError("Missing Polygon API credentials")

logger.info(f"Using Polygon.io for market data")


# Simple Moving Average Crossover Strategy
class SMAStrategy(bt.Strategy):
    """Simple Moving Average Crossover strategy"""
    
    params = (
        ('fast_period', 10),     # Fast moving average period
        ('slow_period', 30),     # Slow moving average period
        ('debug', False),        # Print debug info
    )
    
    def __init__(self):
        # Initialize indicators
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.params.fast_period)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.params.slow_period)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        
        # State variables
        self.order = None
        self.price = None
        self.comm = None
        
        # Record all trades
        self.trades = []
        
        # Debug
        if self.params.debug:
            self.log(f'Strategy initialized with fast_period={self.params.fast_period}, slow_period={self.params.slow_period}')
    
    def log(self, txt):
        """Logging function"""
        dt = self.datas[0].datetime.date(0)
        logger.debug(f'{dt.isoformat()} - {txt}')
    
    def notify_order(self, order):
        """Called on order status changes"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.5f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.price = order.executed.price
                self.comm = order.executed.comm
            else:
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.5f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected: {order.status}')
        
        self.order = None
    
    def notify_trade(self, trade):
        """Called on trade close"""
        if not trade.isclosed:
            return
        
        self.log(f'TRADE CLOSED - Profit: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}')
        
        # Add to trade history - Use string representation for dates
        try:
            # If dtopen/dtclose are datetime objects
            if hasattr(trade.dtopen, 'date'):
                entry_date = trade.dtopen.date().isoformat()
            else:
                # If they're floats, convert to string
                entry_date = str(trade.dtopen)
                
            if hasattr(trade.dtclose, 'date'):
                exit_date = trade.dtclose.date().isoformat()
            else:
                exit_date = str(trade.dtclose)
                
            self.trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': trade.price,
                'exit_price': trade.pclose,
                'size': trade.size,
                'pnl': trade.pnl,
                'pnl_pct': (trade.pnl / trade.price) * 100 if trade.price else 0
            })
        except Exception as e:
            self.log(f"Error recording trade: {e}")
    
    def next(self):
        """Main strategy logic - called for each candle"""
        # Log current close price and indicator values
        if self.params.debug:
            self.log(f'Close: {self.data.close[0]:.5f}, Fast MA: {self.fast_ma[0]:.5f}, Slow MA: {self.slow_ma[0]:.5f}')
        
        # Skip if an order is pending
        if self.order:
            return
        
        # Check for crossover signals
        if not self.position:  # not in the market
            # BUY signal: fast crosses above slow
            if self.crossover > 0:
                self.log(f'BUY SIGNAL - Fast MA crossed above Slow MA')
                self.order = self.buy()
        else:
            # SELL signal: fast crosses below slow
            if self.crossover < 0:
                self.log(f'SELL SIGNAL - Fast MA crossed below Slow MA')
                self.order = self.sell()


class PolygonClient:
    """Client for fetching historical data from Polygon.io"""
    
    def __init__(self, api_key=None):
        """Initialize Polygon client"""
        self.api_key = api_key or POLYGON_API_KEY
        self.base_url = "https://api.polygon.io"
        
        # Check if API key is provided
        if not self.api_key:
            logger.warning("Polygon API key not provided")
            
        logger.info("PolygonClient initialized")
    
    def get_historical_data(self, symbol, start_date, end_date, timespan="day", multiplier=1):
        """
        Fetch historical candle data from Polygon.io
        
        Args:
            symbol: Symbol code (e.g., "EUR/USD" or "EURUSD")
            start_date: Start date as string or datetime
            end_date: End date as string or datetime
            timespan: Timeframe (e.g., "day", "hour", "minute", etc.)
            multiplier: Multiplier for timespan (e.g., 1, 5, 15, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Convert dates to YYYY-MM-DD format
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Format symbol for Polygon (forex symbols need C: prefix)
        polygon_symbol = symbol.replace("/", "").replace("_", "")  # Remove / and _ characters
        if "USD" in polygon_symbol and not polygon_symbol.startswith("C:"):
            polygon_symbol = f"C:{polygon_symbol}"
            
        logger.debug(f"Fetching {multiplier} {timespan} candles for {polygon_symbol} from {start_str} to {end_str}")
        
        # Polygon endpoint for historical data
        url = f"{self.base_url}/v2/aggs/ticker/{polygon_symbol}/range/{multiplier}/{timespan}/{start_str}/{end_str}"
        
        # Parameters for the request
        params = {
            "apiKey": self.api_key,
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000
        }
        
        try:
            # Make the request
            logger.debug(f"API Request: {url}")
            logger.debug(f"Request params: {params}")
            
            response = requests.get(url, params=params)
            logger.debug(f"Response status: {response.status_code}")
            
            # Raise exception for HTTP errors
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            
            # Log abbreviated response
            response_text = json.dumps(data)
            logger.debug(f"Response: {response_text[:500]}{'...' if len(response_text) > 500 else ''}")
            
            # Check if results exist
            if data.get("status") != "OK" or "results" not in data:
                logger.warning(f"No data returned for {polygon_symbol} from {start_str} to {end_str}")
                if "error" in data:
                    logger.error(f"Polygon API error: {data['error']}")
                return pd.DataFrame()
            
            results = data["results"]
            
            # Process results into DataFrame
            ohlc_data = []
            for bar in results:
                # Convert timestamp from milliseconds to datetime
                dt = pd.to_datetime(bar["t"], unit="ms")
                
                ohlc_data.append({
                    "datetime": dt,
                    "open": bar["o"],
                    "high": bar["h"],
                    "low": bar["l"],
                    "close": bar["c"],
                    "volume": bar["v"]
                })
            
            # Create DataFrame
            df = pd.DataFrame(ohlc_data)
            if not df.empty:
                df.set_index("datetime", inplace=True)
            
            logger.info(f"Fetched {len(df)} candles for {polygon_symbol}")
            return df
            
        except requests.exceptions.HTTPError as e:
            # Try to get error details from response
            error_details = {}
            try:
                error_details = response.json()
            except:
                pass
            
            logger.error(f"HTTP Error: {e}")
            logger.error(f"Error details: {error_details}")
            raise
            
        except Exception as e:
            logger.error(f"Request error: {e}")
            raise
    
    def generate_synthetic_data(self, start_date, end_date, volatility=0.005, trend=0.0001):
        """Generate synthetic price data for EUR/USD when API fails"""
        logger.info(f"Generating synthetic data from {start_date} to {end_date}")
        
        # Convert dates to datetime
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Set initial price
        initial_price = 1.10  # Initial EUR/USD price
        
        # Generate prices
        np.random.seed(42)  # For reproducibility
        returns = np.random.normal(trend, volatility, size=len(date_range))
        log_returns = np.cumsum(returns)
        prices = initial_price * np.exp(log_returns)
        
        # Create DataFrame
        df = pd.DataFrame(index=date_range)
        df['open'] = prices * (1 + np.random.normal(0, 0.001, size=len(date_range)))
        df['close'] = prices
        df['high'] = np.maximum(df['open'], df['close']) * (1 + np.random.uniform(0.001, 0.003, size=len(date_range)))
        df['low'] = np.minimum(df['open'], df['close']) * (1 - np.random.uniform(0.001, 0.003, size=len(date_range)))
        df['volume'] = np.random.randint(100000, 10000000, size=len(date_range))
        
        logger.info(f"Generated {len(df)} days of synthetic data")
        return df


class TradingBot:
    """Simple EUR/USD trading bot with backtesting capabilities"""
    
    def __init__(self, instrument="EUR/USD", granularity="day"):
        self.instrument = instrument
        self.granularity = granularity
        self.polygon_client = PolygonClient()
        
        # Strategy parameters
        self.fast_period = 10
        self.slow_period = 30
        
        # Trading parameters
        self.default_units = 1000  # Default position size (use negative for sell)
        
        logger.info(f"Trading bot initialized for {instrument} with {granularity} timeframe")
        logger.info(f"Strategy: SMA Crossover with fast={self.fast_period}, slow={self.slow_period}")
    
    def set_parameters(self, fast_period=10, slow_period=30, units=1000):
        """Set strategy parameters"""
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.default_units = units
        logger.info(f"Parameters set: fast_period={fast_period}, slow_period={slow_period}, units={units}")
    
    def backtest(self, start_date, end_date, initial_cash=10000, commission=0.0001):
        """Run backtest with historical data"""
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        # Map granularity to Polygon timespan
        timespan_map = {
            "minute": "minute",
            "hour": "hour",
            "day": "day",
            "week": "week",
            "month": "month",
            "quarter": "quarter",
            "year": "year",
            "M1": "minute",
            "M5": "minute",
            "M15": "minute",
            "M30": "minute",
            "H1": "hour",
            "H4": "hour",
            "D": "day",
            "W": "week",
            "M": "month"
        }
        
        # Map granularity to Polygon multiplier
        multiplier_map = {
            "minute": 1,
            "hour": 1,
            "day": 1,
            "week": 1,
            "month": 1,
            "quarter": 1,
            "year": 1,
            "M1": 1,
            "M5": 5,
            "M15": 15,
            "M30": 30,
            "H1": 1,
            "H4": 4,
            "D": 1,
            "W": 1,
            "M": 1
        }
        
        # Convert granularity to Polygon format
        timespan = timespan_map.get(self.granularity, "day")
        multiplier = multiplier_map.get(self.granularity, 1)
        
        try:
            # Fetch historical data from Polygon
            df = self.polygon_client.get_historical_data(
                self.instrument,
                start_date,
                end_date,
                timespan=timespan,
                multiplier=multiplier
            )
            
            # If no data from Polygon, use synthetic data
            if df.empty:
                logger.warning("No data from Polygon API, using synthetic data")
                df = self.polygon_client.generate_synthetic_data(start_date, end_date)
        except Exception as e:
            logger.error(f"Error fetching data from Polygon: {e}")
            logger.info("Falling back to synthetic data")
            df = self.polygon_client.generate_synthetic_data(start_date, end_date)
        
        if df.empty:
            logger.error("No data available for backtest")
            return None
        
        # Use a better approach with BacktraderCSVData
        # Create a CSV format that backtrader can directly understand
        date_format = '%Y-%m-%d'
        
        # First create a properly formatted CSV with datetime as the first column
        data_for_bt = []
        for idx, row in df.iterrows():
            date_str = idx.strftime(date_format)
            data_for_bt.append([
                date_str,           # Date in YYYY-MM-DD format
                row['open'],        # Open price
                row['high'],        # High price
                row['low'],         # Low price
                row['close'],       # Close price
                row['volume']       # Volume
            ])
            
        # Save to CSV with proper headers
        csv_path = f"data/{self.instrument.replace('/', '_')}_{start_date}_{end_date}_{self.granularity}.csv"
        with open(csv_path, 'w') as f:
            f.write('datetime,open,high,low,close,volume\n')  # Add header
            for row in data_for_bt:
                f.write(','.join(str(item) for item in row) + '\n')
                
        logger.info(f"Saved properly formatted data to {csv_path}")
        
        # Create cerebro
        cerebro = bt.Cerebro()
        
        # Add strategy
        cerebro.addstrategy(
            SMAStrategy, 
            fast_period=self.fast_period, 
            slow_period=self.slow_period,
            debug=False  # Changed from True to False to reduce log output
        )
        
        # Set broker parameters
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=commission)
        
        # Use a clearer CSV data feed configuration
        data = bt.feeds.GenericCSVData(
            dataname=csv_path,
            dtformat='%Y-%m-%d',
            date=0,           # Column 0 is date
            open=1,           # Column 1 is open
            high=2,           # Column 2 is high
            low=3,            # Column 3 is low
            close=4,          # Column 4 is close
            volume=5,         # Column 5 is volume
            openinterest=-1,  # No open interest
            nullvalue=0.0,    # Replace possible NA values with 0.0
            fromdate=pd.to_datetime(start_date),
            todate=pd.to_datetime(end_date)
        )
        cerebro.adddata(data)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        # Run backtest
        logger.info("Starting backtest")
        results = cerebro.run()
        
        # Get result metrics
        strat = results[0]
        
        # Extract analyzer results
        sharpe_ratio = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
        max_drawdown = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)
        total_return = strat.analyzers.returns.get_analysis().get('rtot', 0)
        trade_analysis = strat.analyzers.trades.get_analysis()
        
        # Calculate win rate
        total_trades = trade_analysis.get('total', {}).get('total', 0)
        won_trades = trade_analysis.get('won', {}).get('total', 0)
        win_rate = won_trades / total_trades if total_trades > 0 else 0
        
        # Log results
        logger.info("Backtest completed")
        logger.info(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {max_drawdown:.2%}")
        logger.info(f"Win Rate: {win_rate:.2%} ({won_trades}/{total_trades})")
        
        # Plot results
        figure_path = f"data/backtest_{self.instrument.replace('/', '_')}_{start_date}_{end_date}.png"
        plt.figure(figsize=(12, 8))
        cerebro.plot(style='candlestick')[0][0]
        plt.savefig(figure_path)
        logger.info(f"Backtest chart saved to {figure_path}")
        
        return {
            'final_value': cerebro.broker.getvalue(),
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'trades': strat.trades
        }


# Main entry point
def main():
    """Main function to run the trading bot"""
    parser = argparse.ArgumentParser(description='EUR/USD Trading Bot with Polygon.io Data')
    parser.add_argument('--mode', choices=['backtest'], default='backtest',
                       help='Trading mode: backtest only (no live mode available)')
    parser.add_argument('--fast', type=int, default=10,
                       help='Fast SMA period')
    parser.add_argument('--slow', type=int, default=30,
                       help='Slow SMA period')
    parser.add_argument('--units', type=int, default=1000,
                       help='Position size in units')
    parser.add_argument('--instrument', type=str, default='EUR/USD',
                       help='Instrument to trade')
    parser.add_argument('--granularity', type=str, default='day',
                       help='Candle granularity (minute, hour, day, week, month)')
    parser.add_argument('--start', type=str, default=(datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d'),
                       help='Start date for backtest (default: 1 year ago)')
    parser.add_argument('--end', type=str, default=datetime.datetime.now().strftime('%Y-%m-%d'),
                       help='End date for backtest (default: today)')
    
    args = parser.parse_args()
    
    # Create trading bot
    bot = TradingBot(instrument=args.instrument, granularity=args.granularity)
    
    # Set parameters
    bot.set_parameters(
        fast_period=args.fast,
        slow_period=args.slow,
        units=args.units
    )
    
    # Run backtest
    logger.info(f"Running backtest from {args.start} to {args.end}")
    results = bot.backtest(args.start, args.end)
    
    if results:
        # Print summary
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Instrument: {args.instrument}")
        print(f"Period: {args.start} to {args.end}")
        print(f"Strategy: SMA Crossover (Fast: {args.fast}, Slow: {args.slow})")
        print(f"Final Portfolio Value: ${results['final_value']:.2f}")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Win Rate: {results['win_rate']:.2%} ({int(results['win_rate'] * results['total_trades'])}/{results['total_trades']})")
        print("="*50)


if __name__ == "__main__":
    try:
        # Print header
        print("="*80)
        print("EUR/USD TRADING BOT WITH POLYGON.IO")
        print("="*80)
        
        # Check environment
        if not POLYGON_API_KEY:
            print("ERROR: Polygon API key not found.")
            print("Please set POLYGON_API_KEY in .env file.")
            exit(1)
        
        print(f"Polygon API Key: {POLYGON_API_KEY[:5]}...{POLYGON_API_KEY[-5:] if len(POLYGON_API_KEY) > 10 else ''}")
        print("\nStarting bot...")
        
        main()
    except KeyboardInterrupt:
        print("\nBot stopped by user. Goodbye!")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\nFatal error: {e}")
