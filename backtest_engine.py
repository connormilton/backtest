#!/usr/bin/env python3
"""
Advanced Modular Backtesting Engine
Supports multiple data sources, strategies, and instruments
Provides quick validation before live trading and strategy optimization
"""

import os
import json
import time
import logging
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backtester.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Backtest_Engine")

# Create output directories
os.makedirs("backtest_results", exist_ok=True)
os.makedirs("backtest_data", exist_ok=True)
os.makedirs("optimization_results", exist_ok=True)

# ======== Data Models ========

@dataclass
class TradeResult:
    """Data class for storing trade results"""
    entry_time: datetime.datetime
    exit_time: Optional[datetime.datetime] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    direction: str = "BUY"  # "BUY" or "SELL"
    size: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    max_drawdown: float = 0.0
    max_profit: float = 0.0
    exit_reason: str = ""  # "tp", "sl", "exit_signal", "manual"
    strategy_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper datetime serialization"""
        result = asdict(self)
        result['entry_time'] = self.entry_time.isoformat() if self.entry_time else None
        result['exit_time'] = self.exit_time.isoformat() if self.exit_time else None
        return result

@dataclass
class BacktestResult:
    """Data class for storing backtest results"""
    start_time: datetime.datetime
    end_time: datetime.datetime
    instrument: str
    strategy_name: str
    initial_balance: float
    final_balance: float
    total_return: float
    total_return_pct: float
    annualized_return: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    consecutive_wins: int
    consecutive_losses: int
    avg_trade_duration: float  # in hours
    trades: List[TradeResult] = field(default_factory=list)
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    parameter_set: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper datetime serialization"""
        result = asdict(self)
        result['start_time'] = self.start_time.isoformat() if self.start_time else None
        result['end_time'] = self.end_time.isoformat() if self.end_time else None
        result['trades'] = [trade.to_dict() for trade in self.trades]
        return result
    
    def save_to_file(self, filename: str = None) -> str:
        """Save backtest results to a JSON file"""
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy_short = self.strategy_name.replace(" ", "_").lower()
            instrument_short = self.instrument.replace("/", "").lower()
            filename = f"backtest_results/{strategy_short}_{instrument_short}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Backtest results saved to {filename}")
        return filename
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'BacktestResult':
        """Create BacktestResult from a dictionary"""
        # Convert string dates back to datetime
        if 'start_time' in data and data['start_time']:
            data['start_time'] = datetime.datetime.fromisoformat(data['start_time'])
        if 'end_time' in data and data['end_time']:
            data['end_time'] = datetime.datetime.fromisoformat(data['end_time'])
        
        # Convert trades
        if 'trades' in data:
            trades = []
            for trade_dict in data['trades']:
                if 'entry_time' in trade_dict and trade_dict['entry_time']:
                    trade_dict['entry_time'] = datetime.datetime.fromisoformat(trade_dict['entry_time'])
                if 'exit_time' in trade_dict and trade_dict['exit_time']:
                    trade_dict['exit_time'] = datetime.datetime.fromisoformat(trade_dict['exit_time'])
                trades.append(TradeResult(**trade_dict))
            data['trades'] = trades
            
        return BacktestResult(**data)
    
    @staticmethod
    def load_from_file(filename: str) -> 'BacktestResult':
        """Load backtest results from a JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        return BacktestResult.from_dict(data)
    
    def plot_equity_curve(self, show: bool = True, save_path: Optional[str] = None) -> None:
        """Plot the equity curve with drawdowns"""
        # Create a balance history from trades
        trade_timestamps = []
        balance_history = []
        drawdown_history = []
        
        current_balance = self.initial_balance
        peak_balance = self.initial_balance
        
        # Start with initial balance
        trade_timestamps.append(self.start_time)
        balance_history.append(current_balance)
        drawdown_history.append(0)
        
        # Add each trade
        for trade in sorted(self.trades, key=lambda t: t.entry_time):
            if trade.exit_time:
                current_balance += trade.pnl
                peak_balance = max(peak_balance, current_balance)
                drawdown = (peak_balance - current_balance) / peak_balance * 100 if peak_balance > 0 else 0
                
                trade_timestamps.append(trade.exit_time)
                balance_history.append(current_balance)
                drawdown_history.append(drawdown)
        
        # Ensure we have at least two points to plot
        if len(trade_timestamps) < 2:
            trade_timestamps.append(self.end_time)
            balance_history.append(current_balance)
            drawdown_history.append(0)
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        
        # Plot equity curve
        ax1.plot(trade_timestamps, balance_history, label='Equity', color='blue', linewidth=2)
        ax1.set_title(f'{self.strategy_name} on {self.instrument} - Equity Curve', fontsize=14)
        ax1.set_ylabel('Account Balance', fontsize=12)
        ax1.grid(True)
        ax1.legend()
        
        # Add annotations for key metrics
        metrics_text = (
            f"Total Return: {self.total_return_pct:.2f}%\n"
            f"Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
            f"Max Drawdown: {self.max_drawdown_pct:.2f}%\n"
            f"Win Rate: {self.win_rate:.2f}%\n"
            f"Total Trades: {self.total_trades}"
        )
        ax1.annotate(metrics_text, xy=(0.02, 0.92), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                    fontsize=10, va='top')
        
        # Plot drawdown
        ax2.fill_between(trade_timestamps, 0, drawdown_history, color='red', alpha=0.3, label='Drawdown')
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True)
        ax2.invert_yaxis()  # Invert so drawdowns go down
        ax2.legend()
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Equity curve saved to {save_path}")
        
        # Display if requested
        if show:
            plt.show()
        else:
            plt.close()

# ======== Data Provider ABC ========

class DataProvider(ABC):
    """
    Abstract base class for data providers
    Implementations can fetch from APIs, CSV files, databases, etc.
    """
    
    @abstractmethod
    def get_historical_data(self, instrument: str, start_date: Union[str, datetime.datetime], 
                           end_date: Union[str, datetime.datetime], timeframe: str = "day") -> pd.DataFrame:
        """
        Fetch historical OHLCV data for the specified instrument and timeframe
        
        Args:
            instrument: Symbol or ticker
            start_date: Start date as string ('YYYY-MM-DD') or datetime
            end_date: End date as string ('YYYY-MM-DD') or datetime
            timeframe: Data timeframe (e.g., "minute", "hour", "day")
            
        Returns:
            DataFrame with OHLCV data and datetime index
        """
        pass
    
    @abstractmethod
    def get_instruments(self) -> List[str]:
        """
        Get list of available instruments
        
        Returns:
            List of instrument symbols/tickers
        """
        pass
    
    @abstractmethod
    def get_latest_data(self, instrument: str, count: int = 1, timeframe: str = "day") -> pd.DataFrame:
        """
        Get the latest N data points for an instrument
        
        Args:
            instrument: Symbol or ticker
            count: Number of latest data points to retrieve
            timeframe: Data timeframe
            
        Returns:
            DataFrame with OHLCV data
        """
        pass

# ======== Strategy ABC ========

class Strategy(ABC):
    """
    Abstract base class for trading strategies
    All custom strategies should inherit from this class
    """
    
    def __init__(self, name: str = "BaseStrategy"):
        """Initialize strategy with name and parameters"""
        self.name = name
        self.parameters = {}
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added signal columns (at minimum, 'signal')
                signal values: 1 (buy), -1 (sell), 0 (no position)
        """
        pass
    
    def set_parameters(self, **kwargs) -> None:
        """Set strategy parameters"""
        self.parameters.update(kwargs)
        
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters"""
        return self.parameters.copy()
    
    def calculate_stop_loss(self, data: pd.DataFrame, signal_row_idx: int, direction: str) -> float:
        """
        Calculate stop loss price for a trade
        
        Args:
            data: DataFrame with OHLCV data
            signal_row_idx: Index of the row with the signal
            direction: 'BUY' or 'SELL'
            
        Returns:
            Stop loss price
        """
        # Default implementation (override in subclasses for custom logic)
        if direction == "BUY":
            # For buy signals, set stop below recent low
            return data.iloc[signal_row_idx]['low'] * 0.99
        else:
            # For sell signals, set stop above recent high
            return data.iloc[signal_row_idx]['high'] * 1.01
    
    def calculate_take_profit(self, data: pd.DataFrame, signal_row_idx: int, direction: str) -> float:
        """
        Calculate take profit price for a trade
        
        Args:
            data: DataFrame with OHLCV data
            signal_row_idx: Index of the row with the signal
            direction: 'BUY' or 'SELL'
            
        Returns:
            Take profit price
        """
        # Default implementation (override in subclasses for custom logic)
        entry_price = data.iloc[signal_row_idx]['close']
        if direction == "BUY":
            # For buy signals, set take profit at 2:1 risk-reward
            stop_loss = self.calculate_stop_loss(data, signal_row_idx, direction)
            risk = entry_price - stop_loss
            return entry_price + (risk * 2)
        else:
            # For sell signals, set take profit at 2:1 risk-reward
            stop_loss = self.calculate_stop_loss(data, signal_row_idx, direction)
            risk = stop_loss - entry_price
            return entry_price - (risk * 2)
    
    def calculate_position_size(self, account_balance: float, risk_per_trade: float, 
                                entry_price: float, stop_loss: float) -> float:
        """
        Calculate position size based on risk parameters
        
        Args:
            account_balance: Current account balance
            risk_per_trade: Risk percentage per trade (e.g., 0.02 for 2%)
            entry_price: Entry price for the trade
            stop_loss: Stop loss price for the trade
            
        Returns:
            Position size in units
        """
        risk_amount = account_balance * risk_per_trade
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit <= 0:
            return 0  # Avoid division by zero
            
        position_size = risk_amount / risk_per_unit
        return position_size

# ======== CSV Data Provider Implementation ========

class CSVDataProvider(DataProvider):
    """Data provider that loads data from CSV files"""
    
    def __init__(self, data_dir: str = "backtest_data"):
        """Initialize with data directory"""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def _get_filename(self, instrument: str, timeframe: str) -> str:
        """Generate a standardized filename for the data"""
        instrument_clean = instrument.replace("/", "_").replace(":", "_")
        return f"{self.data_dir}/{instrument_clean}_{timeframe}.csv"
        
    def save_data(self, data: pd.DataFrame, instrument: str, timeframe: str) -> str:
        """Save data to CSV file"""
        filename = self._get_filename(instrument, timeframe)
        data.to_csv(filename)
        logger.info(f"Saved data for {instrument} ({timeframe}) to {filename}")
        return filename
        
    def get_historical_data(self, instrument: str, start_date: Union[str, datetime.datetime], 
                           end_date: Union[str, datetime.datetime], timeframe: str = "day") -> pd.DataFrame:
        """Fetch historical data from CSV file, filtered by date range"""
        filename = self._get_filename(instrument, timeframe)
        
        if not os.path.exists(filename):
            logger.warning(f"No data file found for {instrument} with {timeframe} timeframe")
            return pd.DataFrame()
            
        # Load data
        df = pd.read_csv(filename, parse_dates=['datetime'])
        df.set_index('datetime', inplace=True)
        
        # Convert date strings to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        # Filter by date range
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        if df.empty:
            logger.warning(f"No data for {instrument} in range {start_date} to {end_date}")
            
        return df
    
    def get_instruments(self) -> List[str]:
        """Get list of available instruments from CSV files"""
        instruments = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.csv'):
                parts = filename.replace('.csv', '').split('_')
                if len(parts) >= 2:
                    # Try to reconstruct the original instrument name
                    if parts[0] in ['C', 'EUR', 'USD', 'GBP', 'JPY', 'AUD', 'CAD', 'NZD', 'CHF']:
                        # Likely a forex pair
                        instrument = f"{parts[0]}/{parts[1]}"
                    else:
                        # Use underscore as separator
                        instrument = '_'.join(parts[:-1])  # Exclude the timeframe part
                    
                    if instrument not in instruments:
                        instruments.append(instrument)
        
        return instruments
    
    def get_latest_data(self, instrument: str, count: int = 1, timeframe: str = "day") -> pd.DataFrame:
        """Get the latest N data points for an instrument"""
        filename = self._get_filename(instrument, timeframe)
        
        if not os.path.exists(filename):
            logger.warning(f"No data file found for {instrument} with {timeframe} timeframe")
            return pd.DataFrame()
            
        # Load data
        df = pd.read_csv(filename, parse_dates=['datetime'])
        df.set_index('datetime', inplace=True)
        
        # Sort by datetime and get latest records
        df = df.sort_index().tail(count)
        
        return df

# ======== Example Strategy Implementations ========

class SMA_Crossover_Strategy(Strategy):
    """Simple Moving Average Crossover Strategy"""
    
    def __init__(self, name: str = "SMA Crossover", fast_period: int = 10, slow_period: int = 30):
        """Initialize with default parameters"""
        super().__init__(name)
        self.parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on SMA crossover"""
        if data.empty:
            return data
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Calculate fast and slow SMAs
        fast_period = self.parameters['fast_period']
        slow_period = self.parameters['slow_period']
        
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
    
    def calculate_stop_loss(self, data: pd.DataFrame, signal_row_idx: int, direction: str) -> float:
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

class RSI_Strategy(Strategy):
    """Relative Strength Index (RSI) Strategy"""
    
    def __init__(self, name: str = "RSI Strategy", rsi_period: int = 14, 
                 oversold: int = 30, overbought: int = 70):
        """Initialize with default parameters"""
        super().__init__(name)
        self.parameters = {
            'rsi_period': rsi_period,
            'oversold': oversold,
            'overbought': overbought
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on RSI indicator"""
        if data.empty:
            return data
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Calculate RSI
        rsi_period = self.parameters['rsi_period']
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
        
        # Calculate RS (Relative Strength)
        rs = gain / loss
        
        # Calculate RSI
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Initialize signal column
        df['signal'] = 0
        
        # Generate buy signals (RSI crosses below oversold threshold and back up)
        oversold = self.parameters['oversold']
        df['prev_rsi'] = df['rsi'].shift(1)
        buy_condition = (df['prev_rsi'] < oversold) & (df['rsi'] > oversold)
        df.loc[buy_condition, 'signal'] = 1
        
        # Generate sell signals (RSI crosses above overbought threshold and back down)
        overbought = self.parameters['overbought']
        sell_condition = (df['prev_rsi'] > overbought) & (df['rsi'] < overbought)
        df.loc[sell_condition, 'signal'] = -1
        
        return df

class Bollinger_Bands_Strategy(Strategy):
    """Bollinger Bands Mean Reversion Strategy"""
    
    def __init__(self, name: str = "Bollinger Bands", bb_period: int = 20, 
                 bb_std: float = 2.0, use_close: bool = True):
        """Initialize with default parameters"""
        super().__init__(name)
        self.parameters = {
            'bb_period': bb_period,
            'bb_std': bb_std,
            'use_close': use_close
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on Bollinger Bands"""
        if data.empty:
            return data
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Calculate Bollinger Bands
        bb_period = self.parameters['bb_period']
        bb_std = self.parameters['bb_std']
        use_close = self.parameters['use_close']
        
        # Calculate middle band (SMA)
        df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
        
        # Calculate standard deviation
        df['bb_std'] = df['close'].rolling(window=bb_period).std()
        
        # Calculate upper and lower bands
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * bb_std)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * bb_std)
        
        # Initialize signal column
        df['signal'] = 0
        
        # Generate buy/sell signals
        price_col = 'close' if use_close else 'low'
        price_high_col = 'close' if use_close else 'high'
        
        # Buy signal: price crosses below lower band and back inside
        df['prev_price'] = df[price_col].shift(1)
        df['prev_lower'] = df['bb_lower'].shift(1)
        buy_condition = (df['prev_price'] < df['prev_lower']) & (df[price_col] > df['bb_lower'])
        df.loc[buy_condition, 'signal'] = 1
        
        # Sell signal: price crosses above upper band and back inside
        df['prev_price_high'] = df[price_high_col].shift(1)
        df['prev_upper'] = df['bb_upper'].shift(1)
        sell_condition = (df['prev_price_high'] > df['prev_upper']) & (df[price_high_col] < df['bb_upper'])
        df.loc[sell_condition, 'signal'] = -1
        
        return df

# ======== Backtester ========

class Backtester:
    """
    Backtesting engine for evaluating trading strategies
    """
    
    def __init__(self, data_provider: DataProvider):
        """Initialize backtester with data provider"""
        self.data_provider = data_provider
    
    def run_backtest(self, strategy: Strategy, instrument: str, 
                    start_date: Union[str, datetime.datetime],
                    end_date: Union[str, datetime.datetime],
                    timeframe: str = "day",
                    initial_balance: float = 10000.0,
                    commission: float = 0.0,
                    slippage: float = 0.0,
                    risk_per_trade: float = 0.02,
                    enable_stop_loss: bool = True,
                    enable_take_profit: bool = True,
                    trade_on_close: bool = True,
                    position_sizing: str = "risk_based",  # "fixed", "risk_based", "percent"
                    fixed_position_size: float = 1.0,
                    percent_balance: float = 0.02,
                    reinvest_profits: bool = True) -> BacktestResult:
        """
        Run a backtest with the specified parameters
        
        Args:
            strategy: Trading strategy (Strategy subclass)
            instrument: Instrument symbol/ticker
            start_date: Start date for backtest
            end_date: End date for backtest
            timeframe: Data timeframe
            initial_balance: Initial account balance
            commission: Commission per trade (percentage)
            slippage: Slippage per trade (percentage)
            risk_per_trade: Risk percentage per trade (for risk-based position sizing)
            enable_stop_loss: Whether to use stop loss
            enable_take_profit: Whether to use take profit
            trade_on_close: Whether to execute trades at close price
            position_sizing: Position sizing method
            fixed_position_size: Fixed position size (for fixed position sizing)
            percent_balance: Percentage of balance to risk (for percent position sizing)
            reinvest_profits: Whether to reinvest profits
            
        Returns:
            BacktestResult object with detailed performance metrics
        """
        # Get historical data
        logger.info(f"Fetching historical data for {instrument} ({timeframe}) from {start_date} to {end_date}")
        data = self.data_provider.get_historical_data(instrument, start_date, end_date, timeframe)
        
        if data.empty:
            logger.error(f"No data available for {instrument} in the specified range")
            return None
        
        # Generate signals
        logger.info(f"Generating signals using {strategy.name}")
        signals_data = strategy.generate_signals(data)
        
        if 'signal' not in signals_data.columns:
            logger.error("Strategy did not generate 'signal' column")
            return None
        
        # Run simulation
        logger.info("Running backtest simulation")
        
        # Initialize variables
        balance = initial_balance
        position = 0  # 0: no position, 1: long, -1: short
        entry_price = 0.0
        entry_time = None
        stop_loss_price = 0.0
        take_profit_price = 0.0
        position_size = 0.0
        trades = []
        max_balance = initial_balance
        min_balance = initial_balance
        current_drawdown = 0.0
        max_drawdown = 0.0
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        daily_returns = []
        
        # Convert dates to datetime objects if they are strings
        if isinstance(start_date, str):
            start_time = pd.to_datetime(start_date)
        else:
            start_time = start_date
            
        if isinstance(end_date, str):
            end_time = pd.to_datetime(end_date)
        else:
            end_time = end_date
        
        # Iterate through data points
        for i in range(1, len(signals_data)):
            current_row = signals_data.iloc[i]
            current_time = signals_data.index[i]
            
            # Skip if missing data
            if pd.isna(current_row['open']) or pd.isna(current_row['high']) or pd.isna(current_row['low']) or pd.isna(current_row['close']):
                continue
            
            # Check for stop loss or take profit (if in a position)
            if position != 0:
                # Calculate entry price with slippage
                adjusted_entry_price = entry_price
                
                # Check for stop loss hit
                if enable_stop_loss and position == 1 and current_row['low'] <= stop_loss_price:
                    # Exit long position at stop loss
                    exit_price = stop_loss_price * (1 - slippage)
                    pnl = (exit_price - adjusted_entry_price) * position_size
                    
                    # Update balance
                    balance += pnl
                    
                    # Record trade
                    trade = TradeResult(
                        entry_time=entry_time,
                        exit_time=current_time,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        direction="BUY",
                        size=position_size,
                        pnl=pnl,
                        pnl_pct=(exit_price / adjusted_entry_price - 1) * 100,
                        stop_loss=stop_loss_price,
                        take_profit=take_profit_price,
                        exit_reason="sl",
                        strategy_name=strategy.name
                    )
                    trades.append(trade)
                    
                    # Reset position
                    position = 0
                    
                    # Update stats
                    if pnl > 0:
                        consecutive_wins += 1
                        consecutive_losses = 0
                    else:
                        consecutive_wins = 0
                        consecutive_losses += 1
                    
                    max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                
                elif enable_stop_loss and position == -1 and current_row['high'] >= stop_loss_price:
                    # Exit short position at stop loss
                    exit_price = stop_loss_price * (1 + slippage)
                    pnl = (adjusted_entry_price - exit_price) * position_size
                    
                    # Update balance
                    balance += pnl
                    
                    # Record trade
                    trade = TradeResult(
                        entry_time=entry_time,
                        exit_time=current_time,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        direction="SELL",
                        size=position_size,
                        pnl=pnl,
                        pnl_pct=(adjusted_entry_price / exit_price - 1) * 100,
                        stop_loss=stop_loss_price,
                        take_profit=take_profit_price,
                        exit_reason="sl",
                        strategy_name=strategy.name
                    )
                    trades.append(trade)
                    
                    # Reset position
                    position = 0
                    
                    # Update stats
                    if pnl > 0:
                        consecutive_wins += 1
                        consecutive_losses = 0
                    else:
                        consecutive_wins = 0
                        consecutive_losses += 1
                    
                    max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                
                # Check for take profit hit
                elif enable_take_profit and position == 1 and current_row['high'] >= take_profit_price:
                    # Exit long position at take profit
                    exit_price = take_profit_price * (1 - slippage)
                    pnl = (exit_price - adjusted_entry_price) * position_size
                    
                    # Update balance
                    balance += pnl
                    
                    # Record trade
                    trade = TradeResult(
                        entry_time=entry_time,
                        exit_time=current_time,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        direction="BUY",
                        size=position_size,
                        pnl=pnl,
                        pnl_pct=(exit_price / adjusted_entry_price - 1) * 100,
                        stop_loss=stop_loss_price,
                        take_profit=take_profit_price,
                        exit_reason="tp",
                        strategy_name=strategy.name
                    )
                    trades.append(trade)
                    
                    # Reset position
                    position = 0
                    
                    # Update stats
                    consecutive_wins += 1
                    consecutive_losses = 0
                    max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                
                elif enable_take_profit and position == -1 and current_row['low'] <= take_profit_price:
                    # Exit short position at take profit
                    exit_price = take_profit_price * (1 + slippage)
                    pnl = (adjusted_entry_price - exit_price) * position_size
                    
                    # Update balance
                    balance += pnl
                    
                    # Record trade
                    trade = TradeResult(
                        entry_time=entry_time,
                        exit_time=current_time,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        direction="SELL",
                        size=position_size,
                        pnl=pnl,
                        pnl_pct=(adjusted_entry_price / exit_price - 1) * 100,
                        stop_loss=stop_loss_price,
                        take_profit=take_profit_price,
                        exit_reason="tp",
                        strategy_name=strategy.name
                    )
                    trades.append(trade)
                    
                    # Reset position
                    position = 0
                    
                    # Update stats
                    consecutive_wins += 1
                    consecutive_losses = 0
                    max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            
            # Check signals for trade entry/exit
            signal = current_row['signal']
            
            # Exit signal (opposite direction of current position)
            if (position == 1 and signal == -1) or (position == -1 and signal == 1):
                # Exit at close or current price depending on settings
                exit_price = current_row['close'] if trade_on_close else current_row['open']
                
                # Apply slippage
                if position == 1:
                    exit_price *= (1 - slippage)
                else:
                    exit_price *= (1 + slippage)
                
                # Calculate PnL
                pnl = 0.0
                if position == 1:
                    pnl = (exit_price - entry_price) * position_size
                else:
                    pnl = (entry_price - exit_price) * position_size
                
                # Apply commission
                pnl -= (position_size * exit_price * commission)
                
                # Update balance
                balance += pnl
                
                # Record trade
                trade = TradeResult(
                    entry_time=entry_time,
                    exit_time=current_time,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    direction="BUY" if position == 1 else "SELL",
                    size=position_size,
                    pnl=pnl,
                    pnl_pct=(exit_price / entry_price - 1) * 100 if position == 1 else (entry_price / exit_price - 1) * 100,
                    stop_loss=stop_loss_price,
                    take_profit=take_profit_price,
                    exit_reason="exit_signal",
                    strategy_name=strategy.name
                )
                trades.append(trade)
                
                # Reset position
                position = 0
                
                # Update stats
                if pnl > 0:
                    consecutive_wins += 1
                    consecutive_losses = 0
                else:
                    consecutive_wins = 0
                    consecutive_losses += 1
                
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            
            # Entry signal (new position)
            elif position == 0 and signal != 0:
                # Enter at close or current price depending on settings
                entry_price = current_row['close'] if trade_on_close else current_row['open']
                
                # Apply slippage
                if signal == 1:
                    entry_price *= (1 + slippage)
                else:
                    entry_price *= (1 - slippage)
                
                # Calculate stop loss and take profit
                if signal == 1:
                    stop_loss_price = strategy.calculate_stop_loss(signals_data, i, "BUY")
                    take_profit_price = strategy.calculate_take_profit(signals_data, i, "BUY")
                else:
                    stop_loss_price = strategy.calculate_stop_loss(signals_data, i, "SELL")
                    take_profit_price = strategy.calculate_take_profit(signals_data, i, "SELL")
                
                # Calculate position size
                if position_sizing == "fixed":
                    position_size = fixed_position_size
                elif position_sizing == "percent":
                    position_size = balance * percent_balance / entry_price
                else:  # risk_based
                    if signal == 1:
                        risk_per_unit = entry_price - stop_loss_price
                    else:
                        risk_per_unit = stop_loss_price - entry_price
                    
                    if risk_per_unit <= 0:
                        continue  # Skip invalid trade
                    
                    risk_amount = balance * risk_per_trade
                    position_size = risk_amount / risk_per_unit
                
                # Apply commission
                balance -= (position_size * entry_price * commission)
                
                # Update position
                position = signal
                entry_time = current_time
            
            # Update performance tracking
            if balance > max_balance:
                max_balance = balance
                current_drawdown = 0
            else:
                current_drawdown = (max_balance - balance) / max_balance if max_balance > 0 else 0
                max_drawdown = max(max_drawdown, current_drawdown)
            
            min_balance = min(min_balance, balance)
            
            # Calculate daily return if this is a new day
            if i > 0 and signals_data.index[i].date() != signals_data.index[i-1].date():
                # Simple return calculation
                if i > 1:
                    prev_balance = balance - (trades[-1].pnl if trades and trades[-1].exit_time.date() == signals_data.index[i].date() else 0)
                    daily_return = (balance / prev_balance) - 1 if prev_balance > 0 else 0
                    daily_returns.append(daily_return)
        
        # Close any open position at the end
        if position != 0:
            # Exit at the last price
            exit_price = signals_data.iloc[-1]['close']
            
            # Apply slippage
            if position == 1:
                exit_price *= (1 - slippage)
            else:
                exit_price *= (1 + slippage)
            
            # Calculate PnL
            pnl = 0.0
            if position == 1:
                pnl = (exit_price - entry_price) * position_size
            else:
                pnl = (entry_price - exit_price) * position_size
            
            # Apply commission
            pnl -= (position_size * exit_price * commission)
            
            # Update balance
            balance += pnl
            
            # Record trade
            trade = TradeResult(
                entry_time=entry_time,
                exit_time=signals_data.index[-1],
                entry_price=entry_price,
                exit_price=exit_price,
                direction="BUY" if position == 1 else "SELL",
                size=position_size,
                pnl=pnl,
                pnl_pct=(exit_price / entry_price - 1) * 100 if position == 1 else (entry_price / exit_price - 1) * 100,
                stop_loss=stop_loss_price,
                take_profit=take_profit_price,
                exit_reason="end_of_backtest",
                strategy_name=strategy.name
            )
            trades.append(trade)
        
        # Calculate performance metrics
        total_trades = len(trades)
        winning_trades = sum(1 for trade in trades if trade.pnl > 0)
        losing_trades = sum(1 for trade in trades if trade.pnl <= 0)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_return = balance - initial_balance
        total_return_pct = (balance / initial_balance - 1) * 100 if initial_balance > 0 else 0
        
        # Calculate time difference in years for annualization
        time_diff = (end_time - start_time).days / 365.25
        annualized_return = ((1 + total_return_pct/100) ** (1/time_diff) - 1) * 100 if time_diff > 0 else 0
        
        # Calculate Sharpe and Sortino ratios
        risk_free_rate = 0.02  # 2% annual risk-free rate
        daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
        
        excess_returns = [r - daily_risk_free for r in daily_returns]
        avg_excess_return = np.mean(excess_returns) if excess_returns else 0
        std_deviation = np.std(excess_returns) if excess_returns else 0
        
        # Avoid division by zero
        sharpe_ratio = (avg_excess_return / std_deviation) * np.sqrt(252) if std_deviation > 0 else 0
        
        # Calculate Sortino ratio (only consider negative returns for downside deviation)
        downside_returns = [r for r in excess_returns if r < 0]
        downside_deviation = np.std(downside_returns) if downside_returns else 0
        sortino_ratio = (avg_excess_return / downside_deviation) * np.sqrt(252) if downside_deviation > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum(trade.pnl for trade in trades if trade.pnl > 0)
        gross_loss = abs(sum(trade.pnl for trade in trades if trade.pnl <= 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate average trade metrics
        avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0
        
        # Calculate best and worst trades
        best_trade = max([trade.pnl for trade in trades]) if trades else 0
        worst_trade = min([trade.pnl for trade in trades]) if trades else 0
        
        # Calculate average trade duration
        durations = [(trade.exit_time - trade.entry_time).total_seconds() / 3600 for trade in trades if trade.exit_time]
        avg_trade_duration = np.mean(durations) if durations else 0
        
        # Calculate monthly returns
        monthly_returns = {}
        if trades:
            for trade in trades:
                if trade.exit_time:
                    month_key = trade.exit_time.strftime('%Y-%m')
                    if month_key not in monthly_returns:
                        monthly_returns[month_key] = 0
                    monthly_returns[month_key] += trade.pnl
        
        # Create result object
        result = BacktestResult(
            start_time=start_time,
            end_time=end_time,
            instrument=instrument,
            strategy_name=strategy.name,
            initial_balance=initial_balance,
            final_balance=balance,
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown * initial_balance,
            max_drawdown_pct=max_drawdown * 100,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            win_rate=win_rate * 100,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            best_trade=best_trade,
            worst_trade=worst_trade,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            consecutive_wins=max_consecutive_wins,
            consecutive_losses=max_consecutive_losses,
            avg_trade_duration=avg_trade_duration,
            trades=trades,
            monthly_returns=monthly_returns,
            parameter_set=strategy.get_parameters()
        )
        
        logger.info(f"Backtest completed for {strategy.name} on {instrument}")
        logger.info(f"Total Return: {total_return_pct:.2f}%, Trades: {total_trades}, Win Rate: {win_rate*100:.2f}%")
        
        return result
    
    def run_parameter_optimization(self, strategy_class, parameter_ranges: Dict[str, List[Any]], 
                                  instrument: str, start_date: Union[str, datetime.datetime],
                                  end_date: Union[str, datetime.datetime], timeframe: str = "day",
                                  optimization_metric: str = "sharpe_ratio",
                                  top_n: int = 5,
                                  **backtest_kwargs) -> List[Tuple[Dict[str, Any], BacktestResult]]:
        """
        Run parameter optimization by testing multiple parameter combinations
        
        Args:
            strategy_class: Strategy class to instantiate
            parameter_ranges: Dictionary of parameter names and their possible values
            instrument: Instrument symbol/ticker
            start_date: Start date for backtest
            end_date: End date for backtest
            timeframe: Data timeframe
            optimization_metric: Metric to optimize for
            top_n: Number of top results to return
            backtest_kwargs: Additional arguments for run_backtest
            
        Returns:
            List of (parameter_set, backtest_result) tuples, sorted by optimization metric
        """
        logger.info(f"Starting parameter optimization for {strategy_class.__name__} on {instrument}")
        
        # Generate parameter combinations
        def generate_parameter_combinations(param_ranges, current_params=None, param_names=None):
            if current_params is None:
                current_params = {}
                param_names = list(param_ranges.keys())
                
            if not param_names:
                yield current_params.copy()
                return
                
            current_param = param_names[0]
            remaining_params = param_names[1:]
            
            for value in param_ranges[current_param]:
                current_params[current_param] = value
                yield from generate_parameter_combinations(param_ranges, current_params, remaining_params)
        
        # Get all parameter combinations
        parameter_combinations = list(generate_parameter_combinations(parameter_ranges))
        logger.info(f"Testing {len(parameter_combinations)} parameter combinations")
        
        # Run backtest for each parameter combination
        results = []
        for i, params in enumerate(parameter_combinations):
            logger.info(f"Testing combination {i+1}/{len(parameter_combinations)}: {params}")
            
            # Create strategy instance with current parameters
            strategy = strategy_class()
            strategy.set_parameters(**params)
            
            # Run backtest
            backtest_result = self.run_backtest(
                strategy=strategy,
                instrument=instrument,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
                **backtest_kwargs
            )
            
            # Store result if valid
            if backtest_result:
                results.append((params, backtest_result))
        
        # Sort results by optimization metric
        metric_getter = lambda x: getattr(x[1], optimization_metric, 0)
        
        # Handle special case for drawdown (lower is better)
        if optimization_metric in ['max_drawdown', 'max_drawdown_pct']:
            sorted_results = sorted(results, key=metric_getter)
        else:
            sorted_results = sorted(results, key=metric_getter, reverse=True)
        
        # Get top N results
        top_results = sorted_results[:top_n]
        
        # Log top results
        logger.info(f"Top {len(top_results)} parameter combinations:")
        for i, (params, result) in enumerate(top_results):
            logger.info(f"#{i+1}: {params} - {optimization_metric}: {metric_getter((params, result))}")
        
        return top_results
    
    def run_walk_forward_analysis(self, strategy_class, parameter_ranges: Dict[str, List[Any]],
                                 instrument: str, start_date: Union[str, datetime.datetime],
                                 end_date: Union[str, datetime.datetime], timeframe: str = "day",
                                 train_size: int = 180, test_size: int = 90,
                                 optimization_metric: str = "sharpe_ratio",
                                 **backtest_kwargs) -> BacktestResult:
        """
        Run walk-forward analysis to prevent overfitting
        
        Args:
            strategy_class: Strategy class to instantiate
            parameter_ranges: Dictionary of parameter names and their possible values
            instrument: Instrument symbol/ticker
            start_date: Start date for backtest
            end_date: End date for backtest
            timeframe: Data timeframe
            train_size: Number of days for training period
            test_size: Number of days for testing period
            optimization_metric: Metric to optimize for
            backtest_kwargs: Additional arguments for run_backtest
            
        Returns:
            Aggregated backtest result across all out-of-sample periods
        """
        logger.info(f"Starting walk-forward analysis for {strategy_class.__name__} on {instrument}")
        
        # Convert dates to datetime objects
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Get full data
        data = self.data_provider.get_historical_data(instrument, start_date, end_date, timeframe)
        if data.empty:
            logger.error(f"No data available for {instrument} in the specified range")
            return None
        
        # Calculate number of windows
        total_days = (end_date - start_date).days
        window_size = train_size + test_size
        n_windows = max(1, total_days // test_size - (train_size // test_size))
        
        logger.info(f"Running {n_windows} walk-forward windows")
        
        # Prepare for walk-forward iterations
        all_oos_trades = []
        all_oos_results = []
        
        # Iterate through walk-forward windows
        for i in range(n_windows):
            # Calculate window dates
            window_start = start_date + datetime.timedelta(days=i*test_size)
            train_end = window_start + datetime.timedelta(days=train_size)
            test_end = min(train_end + datetime.timedelta(days=test_size), end_date)
            
            logger.info(f"Window {i+1}/{n_windows}:")
            logger.info(f"  Train: {window_start} to {train_end}")
            logger.info(f"  Test: {train_end} to {test_end}")
            
            # Skip if window extends beyond available data
            if window_start >= end_date or train_end >= end_date:
                logger.warning(f"Window {i+1} extends beyond end date, skipping")
                continue
            
            # Optimize parameters on training data
            top_params = self.run_parameter_optimization(
                strategy_class=strategy_class,
                parameter_ranges=parameter_ranges,
                instrument=instrument,
                start_date=window_start,
                end_date=train_end,
                timeframe=timeframe,
                optimization_metric=optimization_metric,
                top_n=1,
                **backtest_kwargs
            )
            
            if not top_params:
                logger.warning(f"No valid parameters found for window {i+1}, skipping")
                continue
            
            # Extract best parameters
            best_params, _ = top_params[0]
            logger.info(f"  Best parameters: {best_params}")
            
            # Test on out-of-sample data
            strategy = strategy_class()
            strategy.set_parameters(**best_params)
            
            oos_result = self.run_backtest(
                strategy=strategy,
                instrument=instrument,
                start_date=train_end,
                end_date=test_end,
                timeframe=timeframe,
                **backtest_kwargs
            )
            
            if oos_result:
                logger.info(f"  OOS Performance: {optimization_metric}={getattr(oos_result, optimization_metric, 0)}")
                
                # Store OOS trades and results
                all_oos_trades.extend(oos_result.trades)
                all_oos_results.append(oos_result)
        
        # Combine all OOS results into a single result
        if not all_oos_results:
            logger.error("No valid out-of-sample results generated")
            return None
        
        # Calculate combined metrics
        total_return = sum(result.total_return for result in all_oos_results)
        initial_balance = all_oos_results[0].initial_balance
        final_balance = initial_balance + total_return
        
        # Calculate performance metrics from combined trades
        total_trades = len(all_oos_trades)
        winning_trades = sum(1 for trade in all_oos_trades if trade.pnl > 0)
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate combined max drawdown
        combined_balances = []
        current_balance = initial_balance
        
        # Sort trades by exit time
        sorted_trades = sorted(all_oos_trades, key=lambda t: t.exit_time if t.exit_time else datetime.datetime.max)
        
        for trade in sorted_trades:
            current_balance += trade.pnl
            combined_balances.append(current_balance)
        
        if combined_balances:
            # Calculate drawdown series
            peak = initial_balance
            drawdowns = []
            
            for balance in combined_balances:
                peak = max(peak, balance)
                drawdown = (peak - balance) / peak if peak > 0 else 0
                drawdowns.append(drawdown)
            
            max_drawdown = max(drawdowns) if drawdowns else 0
        else:
            max_drawdown = 0
        
        # Create combined result
        combined_result = BacktestResult(
            start_time=all_oos_results[0].start_time,
            end_time=all_oos_results[-1].end_time,
            instrument=instrument,
            strategy_name=strategy_class.__name__,
            initial_balance=initial_balance,
            final_balance=final_balance,
            total_return=total_return,
            total_return_pct=(final_balance / initial_balance - 1) * 100 if initial_balance > 0 else 0,
            annualized_return=0,  # Will be calculated below
            max_drawdown=max_drawdown * initial_balance,
            max_drawdown_pct=max_drawdown * 100,
            sharpe_ratio=0,  # Will be calculated below
            sortino_ratio=0,  # Will be calculated below
            win_rate=win_rate * 100,
            profit_factor=0,  # Will be calculated below
            avg_win=sum(trade.pnl for trade in all_oos_trades if trade.pnl > 0) / winning_trades if winning_trades > 0 else 0,
            avg_loss=sum(abs(trade.pnl) for trade in all_oos_trades if trade.pnl <= 0) / losing_trades if losing_trades > 0 else 0,
            best_trade=max([trade.pnl for trade in all_oos_trades]) if all_oos_trades else 0,
            worst_trade=min([trade.pnl for trade in all_oos_trades]) if all_oos_trades else 0,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            consecutive_wins=0,  # Not tracked across OOS periods
            consecutive_losses=0,  # Not tracked across OOS periods
            avg_trade_duration=np.mean([(trade.exit_time - trade.entry_time).total_seconds() / 3600 
                                     for trade in all_oos_trades if trade.exit_time]) if all_oos_trades else 0,
            trades=all_oos_trades.copy(),
            monthly_returns={},  # Will be calculated below
            parameter_set={},  # Multiple parameter sets used
            metadata={"walk_forward": True, "n_windows": n_windows}
        )
        
        # Calculate annualized return
        total_days = (combined_result.end_time - combined_result.start_time).days
        years = total_days / 365.25
        combined_result.annualized_return = ((1 + combined_result.total_return_pct/100) ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum(trade.pnl for trade in all_oos_trades if trade.pnl > 0)
        gross_loss = abs(sum(trade.pnl for trade in all_oos_trades if trade.pnl <= 0))
        combined_result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate monthly returns
        monthly_returns = {}
        for trade in all_oos_trades:
            if trade.exit_time:
                month_key = trade.exit_time.strftime('%Y-%m')
                if month_key not in monthly_returns:
                    monthly_returns[month_key] = 0
                monthly_returns[month_key] += trade.pnl
        
        combined_result.monthly_returns = monthly_returns
        
        # Calculate Sharpe and Sortino ratios
        if monthly_returns:
            monthly_return_values = list(monthly_returns.values())
            monthly_return_pct = [r / initial_balance for r in monthly_return_values]
            
            risk_free_rate = 0.02  # 2% annual risk-free rate
            monthly_risk_free = (1 + risk_free_rate) ** (1/12) - 1
            
            excess_returns = [r - monthly_risk_free for r in monthly_return_pct]
            avg_excess_return = np.mean(excess_returns) if excess_returns else 0
            std_deviation = np.std(excess_returns) if excess_returns else 0
            
            # Avoid division by zero
            combined_result.sharpe_ratio = (avg_excess_return / std_deviation) * np.sqrt(12) if std_deviation > 0 else 0
            
            # Calculate Sortino ratio (only consider negative returns for downside deviation)
            downside_returns = [r for r in excess_returns if r < 0]
            downside_deviation = np.std(downside_returns) if downside_returns else 0
            combined_result.sortino_ratio = (avg_excess_return / downside_deviation) * np.sqrt(12) if downside_deviation > 0 else 0
        
        logger.info(f"Walk-forward analysis completed with {total_trades} trades and {win_rate*100:.2f}% win rate")
        
        return combined_result
    
    def run_monte_carlo_simulation(self, backtest_result: BacktestResult, 
                                  n_simulations: int = 1000,
                                  confidence_interval: float = 0.95) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation to analyze the robustness of a strategy
        
        Args:
            backtest_result: Result of a backtest run
            n_simulations: Number of Monte Carlo simulations to run
            confidence_interval: Confidence interval for the results (0-1)
            
        Returns:
            Dictionary with simulation results
        """
        logger.info(f"Running Monte Carlo simulation with {n_simulations} iterations")
        
        if not backtest_result or not backtest_result.trades:
            logger.error("No trades available for Monte Carlo simulation")
            return None
        
        # Extract trade returns
        trade_returns = [trade.pnl_pct / 100 for trade in backtest_result.trades]
        
        # Run simulations
        sim_equity_curves = []
        sim_max_drawdowns = []
        sim_final_equities = []
        
        for _ in range(n_simulations):
            # Randomly sample trades with replacement
            sampled_returns = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
            
            # Calculate equity curve
            equity_curve = [1.0]
            for ret in sampled_returns:
                equity_curve.append(equity_curve[-1] * (1 + ret))
            
            # Calculate max drawdown
            peak = equity_curve[0]
            drawdowns = []
            
            for equity in equity_curve:
                peak = max(peak, equity)
                drawdown = (peak - equity) / peak if peak > 0 else 0
                drawdowns.append(drawdown)
            
            max_drawdown = max(drawdowns)
            
            # Store results
            sim_equity_curves.append(equity_curve)
            sim_max_drawdowns.append(max_drawdown * 100)  # As percentage
            sim_final_equities.append(equity_curve[-1])
        
        # Calculate confidence intervals
        lower_bound = (1 - confidence_interval) / 2
        upper_bound = 1 - lower_bound
        
        final_equity_lower = np.percentile(sim_final_equities, lower_bound * 100)
        final_equity_upper = np.percentile(sim_final_equities, upper_bound * 100)
        final_equity_median = np.median(sim_final_equities)
        
        drawdown_lower = np.percentile(sim_max_drawdowns, lower_bound * 100)
        drawdown_upper = np.percentile(sim_max_drawdowns, upper_bound * 100)
        drawdown_median = np.median(sim_max_drawdowns)
        
        # Calculate probability of loss
        prob_loss = sum(1 for equity in sim_final_equities if equity < 1) / n_simulations
        
        # Calculate probability of 10% or greater drawdown
        prob_10pct_drawdown = sum(1 for dd in sim_max_drawdowns if dd >= 10) / n_simulations
        
        # Calculate expected return
        expected_return = (np.mean(sim_final_equities) - 1) * 100
        
        # Calculate potential worst case scenario (99th percentile drawdown)
        worst_case_drawdown = np.percentile(sim_max_drawdowns, 99)
        
        # Create result dictionary
        result = {
            "n_simulations": n_simulations,
            "confidence_interval": confidence_interval,
            "final_equity": {
                "median": final_equity_median,
                "lower_bound": final_equity_lower,
                "upper_bound": final_equity_upper,
                "expected_return_pct": expected_return
            },
            "max_drawdown_pct": {
                "median": drawdown_median,
                "lower_bound": drawdown_lower,
                "upper_bound": drawdown_upper,
                "worst_case": worst_case_drawdown
            },
            "probabilities": {
                "loss": prob_loss,
                "drawdown_10pct_or_greater": prob_10pct_drawdown
            }
        }
        
        logger.info(f"Monte Carlo simulation completed")
        logger.info(f"Expected return: {expected_return:.2f}%")
        logger.info(f"Expected max drawdown: {drawdown_median:.2f}% ({drawdown_lower:.2f}% - {drawdown_upper:.2f}%)")
        logger.info(f"Probability of loss: {prob_loss*100:.2f}%")
        
        return result
    
    def generate_report(self, backtest_result: BacktestResult, include_monte_carlo: bool = True,
                       n_simulations: int = 1000) -> None:
        """
        Generate a comprehensive HTML report for backtest results
        
        Args:
            backtest_result: Result of a backtest run
            include_monte_carlo: Whether to include Monte Carlo simulation
            n_simulations: Number of Monte Carlo simulations to run
        """
        # Generate report file name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"backtest_results/report_{backtest_result.strategy_name}_{backtest_result.instrument.replace('/', '')}_{timestamp}.html"
        
        # Run Monte Carlo simulation if requested
        monte_carlo_results = None
        if include_monte_carlo and len(backtest_result.trades) > 5:
            monte_carlo_results = self.run_monte_carlo_simulation(backtest_result, n_simulations)
        
        # Generate plots and save them
        plot_dir = "backtest_results/plots"
        os.makedirs(plot_dir, exist_ok=True)
        
        # Equity curve plot
        equity_plot = f"{plot_dir}/equity_{timestamp}.png"
        backtest_result.plot_equity_curve(show=False, save_path=equity_plot)
        
        # Monthly returns heatmap
        monthly_returns_plot = f"{plot_dir}/monthly_returns_{timestamp}.png"
        self._plot_monthly_returns_heatmap(backtest_result, save_path=monthly_returns_plot)
        
        # Plot trade distribution
        trade_distribution_plot = f"{plot_dir}/trade_distribution_{timestamp}.png"
        self._plot_trade_distribution(backtest_result, save_path=trade_distribution_plot)
        
        # Create HTML report
        with open(report_filename, 'w') as f:
            f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report: {backtest_result.strategy_name} on {backtest_result.instrument}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .summary {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }}
        .metric {{ flex: 1; min-width: 200px; padding: 15px; background-color: #f9f9f9; border-radius: 5px; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .plot {{ margin: 20px 0; max-width: 100%; }}
        .plot img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Backtest Report</h1>
        <h2>{backtest_result.strategy_name} on {backtest_result.instrument}</h2>
        <p>Period: {backtest_result.start_time.strftime('%Y-%m-%d')} to {backtest_result.end_time.strftime('%Y-%m-%d')}</p>
        
        <h3>Summary</h3>
        <div class="summary">
            <div class="metric">
                <h4>Total Return</h4>
                <p class="{self._get_class(backtest_result.total_return_pct)}">{backtest_result.total_return_pct:.2f}%</p>
            </div>
            <div class="metric">
                <h4>Annual Return</h4>
                <p class="{self._get_class(backtest_result.annualized_return)}">{backtest_result.annualized_return:.2f}%</p>
            </div>
            <div class="metric">
                <h4>Sharpe Ratio</h4>
                <p class="{self._get_class(backtest_result.sharpe_ratio)}">{backtest_result.sharpe_ratio:.2f}</p>
            </div>
            <div class="metric">
                <h4>Sortino Ratio</h4>
                <p class="{self._get_class(backtest_result.sortino_ratio)}">{backtest_result.sortino_ratio:.2f}</p>
            </div>
            <div class="metric">
                <h4>Max Drawdown</h4>
                <p class="{self._get_class(-backtest_result.max_drawdown_pct)}">{backtest_result.max_drawdown_pct:.2f}%</p>
            </div>
            <div class="metric">
                <h4>Win Rate</h4>
                <p class="{self._get_class(backtest_result.win_rate - 50)}">{backtest_result.win_rate:.2f}%</p>
            </div>
            <div class="metric">
                <h4>Profit Factor</h4>
                <p class="{self._get_class(backtest_result.profit_factor - 1)}">{backtest_result.profit_factor:.2f}</p>
            </div>
            <div class="metric">
                <h4>Total Trades</h4>
                <p>{backtest_result.total_trades}</p>
            </div>
        </div>
        
        <h3>Performance Metrics</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Initial Balance</td>
                <td>${backtest_result.initial_balance:.2f}</td>
            </tr>
            <tr>
                <td>Final Balance</td>
                <td>${backtest_result.final_balance:.2f}</td>
            </tr>
            <tr>
                <td>Total Return</td>
                <td>${backtest_result.total_return:.2f} ({backtest_result.total_return_pct:.2f}%)</td>
            </tr>
            <tr>
                <td>Annualized Return</td>
                <td>{backtest_result.annualized_return:.2f}%</td>
            </tr>
            <tr>
                <td>Max Drawdown</td>
                <td>${backtest_result.max_drawdown:.2f} ({backtest_result.max_drawdown_pct:.2f}%)</td>
            </tr>
            <tr>
                <td>Sharpe Ratio</td>
                <td>{backtest_result.sharpe_ratio:.2f}</td>
            </tr>
            <tr>
                <td>Sortino Ratio</td>
                <td>{backtest_result.sortino_ratio:.2f}</td>
            </tr>
            <tr>
                <td>Win Rate</td>
                <td>{backtest_result.win_rate:.2f}%</td>
            </tr>
            <tr>
                <td>Profit Factor</td>
                <td>{backtest_result.profit_factor:.2f}</td>
            </tr>
            <tr>
                <td>Average Win</td>
                <td>${backtest_result.avg_win:.2f}</td>
            </tr>
            <tr>
                <td>Average Loss</td>
                <td>${backtest_result.avg_loss:.2f}</td>
            </tr>
            <tr>
                <td>Best Trade</td>
                <td>${backtest_result.best_trade:.2f}</td>
            </tr>
            <tr>
                <td>Worst Trade</td>
                <td>${backtest_result.worst_trade:.2f}</td>
            </tr>
            <tr>
                <td>Total Trades</td>
                <td>{backtest_result.total_trades}</td>
            </tr>
            <tr>
                <td>Winning Trades</td>
                <td>{backtest_result.winning_trades}</td>
            </tr>
            <tr>
                <td>Losing Trades</td>
                <td>{backtest_result.losing_trades}</td>
            </tr>
            <tr>
                <td>Max Consecutive Wins</td>
                <td>{backtest_result.consecutive_wins}</td>
            </tr>
            <tr>
                <td>Max Consecutive Losses</td>
                <td>{backtest_result.consecutive_losses}</td>
            </tr>
            <tr>
                <td>Average Trade Duration</td>
                <td>{backtest_result.avg_trade_duration:.2f} hours</td>
            </tr>
        </table>
        
        <h3>Strategy Parameters</h3>
        <table>
            <tr>
                <th>Parameter</th>
                <th>Value</th>
            </tr>
""")
            # Add strategy parameters
            for param, value in backtest_result.parameter_set.items():
                f.write(f"            <tr>\n                <td>{param}</td>\n                <td>{value}</td>\n            </tr>\n")
            
            f.write(f"""        </table>
        
        <h3>Equity Curve</h3>
        <div class="plot">
            <img src="{equity_plot}" alt="Equity Curve">
        </div>
        
        <h3>Monthly Returns Heatmap</h3>
        <div class="plot">
            <img src="{monthly_returns_plot}" alt="Monthly Returns Heatmap">
        </div>
        
        <h3>Trade Distribution</h3>
        <div class="plot">
            <img src="{trade_distribution_plot}" alt="Trade Distribution">
        </div>
""")
            
            # Add Monte Carlo section if available
            if monte_carlo_results:
                f.write(f"""
        <h3>Monte Carlo Simulation ({monte_carlo_results['n_simulations']} simulations)</h3>
        <h4>Final Equity</h4>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Expected Return</td>
                <td>{monte_carlo_results['final_equity']['expected_return_pct']:.2f}%</td>
            </tr>
            <tr>
                <td>Median Final Equity</td>
                <td>{monte_carlo_results['final_equity']['median']:.4f}</td>
            </tr>
            <tr>
                <td>{monte_carlo_results['confidence_interval']*100:.0f}% Confidence Interval</td>
                <td>{monte_carlo_results['final_equity']['lower_bound']:.4f} to {monte_carlo_results['final_equity']['upper_bound']:.4f}</td>
            </tr>
        </table>
        
        <h4>Maximum Drawdown</h4>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Median Max Drawdown</td>
                <td>{monte_carlo_results['max_drawdown_pct']['median']:.2f}%</td>
            </tr>
            <tr>
                <td>{monte_carlo_results['confidence_interval']*100:.0f}% Confidence Interval</td>
                <td>{monte_carlo_results['max_drawdown_pct']['lower_bound']:.2f}% to {monte_carlo_results['max_drawdown_pct']['upper_bound']:.2f}%</td>
            </tr>
            <tr>
                <td>Worst Case Scenario (99%)</td>
                <td>{monte_carlo_results['max_drawdown_pct']['worst_case']:.2f}%</td>
            </tr>
        </table>
        
        <h4>Probabilities</h4>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Probability of Loss</td>
                <td>{monte_carlo_results['probabilities']['loss']*100:.2f}%</td>
            </tr>
            <tr>
                <td>Probability of 10%+ Drawdown</td>
                <td>{monte_carlo_results['probabilities']['drawdown_10pct_or_greater']*100:.2f}%</td>
            </tr>
        </table>
""")
            
            # Add trade list table
            f.write(f"""
        <h3>Trade List</h3>
        <table>
            <tr>
                <th>#</th>
                <th>Entry Date</th>
                <th>Exit Date</th>
                <th>Direction</th>
                <th>Entry Price</th>
                <th>Exit Price</th>
                <th>P&L</th>
                <th>P&L %</th>
                <th>Exit Reason</th>
            </tr>
""")
            # Add trade rows
            for i, trade in enumerate(backtest_result.trades):
                entry_date = trade.entry_time.strftime('%Y-%m-%d %H:%M') if trade.entry_time else 'N/A'
                exit_date = trade.exit_time.strftime('%Y-%m-%d %H:%M') if trade.exit_time else 'N/A'
                pnl_class = "positive" if trade.pnl > 0 else "negative"
                f.write(f"""            <tr>
                <td>{i+1}</td>
                <td>{entry_date}</td>
                <td>{exit_date}</td>
                <td>{trade.direction}</td>
                <td>{trade.entry_price:.5f}</td>
                <td>{trade.exit_price:.5f}</td>
                <td class="{pnl_class}">${trade.pnl:.2f}</td>
                <td class="{pnl_class}">{trade.pnl_pct:.2f}%</td>
                <td>{trade.exit_reason}</td>
            </tr>
""")
            
            # Close HTML tags
            f.write("""        </table>
    </div>
</body>
</html>""")
        
        logger.info(f"Backtest report generated: {report_filename}")
        return report_filename
    
    def _get_class(self, value):
        """Helper method to determine CSS class based on value"""
        if value > 0:
            return "positive"
        elif value < 0:
            return "negative"
        else:
            return ""
    
    def _plot_monthly_returns_heatmap(self, backtest_result: BacktestResult, save_path: str = None) -> None:
        """Plot monthly returns as a heatmap"""
        if not backtest_result.monthly_returns:
            logger.warning("No monthly returns data available for heatmap")
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "No monthly returns data available", 
                    horizontalalignment='center', verticalalignment='center')
            if save_path:
                plt.savefig(save_path)
            plt.close()
            return
        
        # Convert monthly returns to a DataFrame with year-month structure
        monthly_data = {}
        for month_key, value in backtest_result.monthly_returns.items():
            year, month = month_key.split('-')
            year = int(year)
            month = int(month)
            
            if year not in monthly_data:
                monthly_data[year] = {}
            
            # Convert to percentage return
            monthly_data[year][month] = (value / backtest_result.initial_balance) * 100
        
        # Get range of years and create DataFrame
        years = sorted(monthly_data.keys())
        months = list(range(1, 13))
        
        # Create empty DataFrame with years as index and months as columns
        df = pd.DataFrame(index=years, columns=months)
        
        # Fill DataFrame with returns
        for year in years:
            for month in months:
                if month in monthly_data[year]:
                    df.loc[year, month] = monthly_data[year][month]
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        
        # Create a mask for NaN values
        mask = df.isna()
        
        # Define colormap with centered zero
        cmap = plt.cm.RdYlGn
        norm = plt.Normalize(-max(abs(df.min().min()), abs(df.max().max())), 
                           max(abs(df.min().min()), abs(df.max().max())))
        
        # Plot heatmap with masked values
        heatmap = ax.pcolor(df.columns, df.index, df.values, cmap=cmap, norm=norm,
                           mask=mask.values)
        
        # Create colorbar
        cbar = plt.colorbar(heatmap)
        cbar.set_label('Monthly Return (%)')
        
        # Format plot
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticks(np.arange(len(months)) + 0.5)
        ax.set_xticklabels(month_names)
        
        ax.set_yticks(np.arange(len(years)) + 0.5)
        ax.set_yticklabels(years)
        
        plt.title(f'Monthly Returns (%) - {backtest_result.strategy_name} on {backtest_result.instrument}')
        
        # Add text annotations
        for i, year in enumerate(years):
            for j, month in enumerate(months):
                if not mask.values[i, j]:
                    value = df.values[i, j]
                    color = 'white' if abs(value) > 10 else 'black'
                    plt.text(j + 0.5, i + 0.5, f'{value:.1f}%',
                           ha="center", va="center", color=color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def _plot_trade_distribution(self, backtest_result: BacktestResult, save_path: str = None) -> None:
        """Plot trade distribution histogram"""
        if not backtest_result.trades:
            logger.warning("No trades available for distribution plot")
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "No trades available", 
                    horizontalalignment='center', verticalalignment='center')
            if save_path:
                plt.savefig(save_path)
            plt.close()
            return
        
        # Extract trade results
        trade_pnls = [trade.pnl for trade in backtest_result.trades]
        trade_pnls_pct = [trade.pnl_pct for trade in backtest_result.trades]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot absolute P&L distribution
        ax1.hist(trade_pnls, bins=20, color='blue', alpha=0.7)
        ax1.axvline(x=0, color='black', linestyle='--')
        ax1.set_title('Trade P&L Distribution')
        ax1.set_xlabel('P&L ($)')
        ax1.set_ylabel('Frequency')
        
        # Plot percentage P&L distribution
        ax2.hist(trade_pnls_pct, bins=20, color='green', alpha=0.7)
        ax2.axvline(x=0, color='black', linestyle='--')
        ax2.set_title('Trade P&L Distribution (%)')
        ax2.set_xlabel('P&L (%)')
        ax2.set_ylabel('Frequency')
        
        # Add statistics as text
        winning_trades = sum(1 for pnl in trade_pnls if pnl > 0)
        losing_trades = sum(1 for pnl in trade_pnls if pnl <= 0)
        win_rate = winning_trades / len(trade_pnls) if trade_pnls else 0
        
        avg_win = sum(pnl for pnl in trade_pnls if pnl > 0) / winning_trades if winning_trades > 0 else 0
        avg_loss = sum(abs(pnl) for pnl in trade_pnls if pnl <= 0) / losing_trades if losing_trades > 0 else 0
        
        stats_text = (
            f"Trades: {len(trade_pnls)}\n"
            f"Win Rate: {win_rate*100:.1f}%\n"
            f"Avg Win: ${avg_win:.2f}\n"
            f"Avg Loss: ${avg_loss:.2f}\n"
            f"Ratio: {avg_win/avg_loss:.2f}" if avg_loss > 0 else "Ratio: ∞"
        )
        
        ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()

# ======== Polygon Data Provider Implementation ========

class PolygonDataProvider(DataProvider):
    """Data provider for Polygon.io API"""
    
    def __init__(self, api_key: str, data_dir: str = "backtest_data"):
        """Initialize with API key"""
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.data_dir = data_dir
        self.csv_provider = CSVDataProvider(data_dir)
        
        # Create session for requests
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}"
        })
        
        os.makedirs(data_dir, exist_ok=True)
    
    def get_historical_data(self, instrument: str, start_date: Union[str, datetime.datetime], 
                           end_date: Union[str, datetime.datetime], timeframe: str = "day") -> pd.DataFrame:
        """Fetch historical OHLCV data for the specified instrument and timeframe"""
        # First check if data is available in CSV
        csv_data = self.csv_provider.get_historical_data(instrument, start_date, end_date, timeframe)
        
        if not csv_data.empty:
            logger.info(f"Using cached data for {instrument} ({timeframe})")
            return csv_data
        
        # Convert dates to string format
        if isinstance(start_date, datetime.datetime):
            start_str = start_date.strftime("%Y-%m-%d")
        else:
            start_str = start_date
            
        if isinstance(end_date, datetime.datetime):
            end_str = end_date.strftime("%Y-%m-%d")
        else:
            end_str = end_date
        
        # Format symbol for Polygon API
        if '/' in instrument:
            ticker = f"C:{instrument.replace('/', '')}"
        else:
            ticker = instrument
        
        # Map timeframe to Polygon format
        timeframe_map = {
            "minute": "minute",
            "hour": "hour",
            "day": "day",
            "week": "week",
            "month": "month"
        }
        
        polygon_timeframe = timeframe_map.get(timeframe, timeframe)
        
        # Set default multiplier
        multiplier = 1
        
        # Handle special cases like M15, H4, etc.
        if timeframe.startswith('M') and len(timeframe) > 1:
            polygon_timeframe = 'minute'
            try:
                multiplier = int(timeframe[1:])
            except ValueError:
                logger.warning(f"Could not parse timeframe: {timeframe}, using default")
        
        if timeframe.startswith('H') and len(timeframe) > 1:
            polygon_timeframe = 'hour'
            try:
                multiplier = int(timeframe[1:])
            except ValueError:
                logger.warning(f"Could not parse timeframe: {timeframe}, using default")
        
        logger.info(f"Fetching {multiplier} {polygon_timeframe} data for {ticker} from {start_str} to {end_str}")
        
        # Build API endpoint
        endpoint = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/{multiplier}/{polygon_timeframe}/{start_str}/{end_str}"
        
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000
        }
        
        try:
            # Make API request
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check if results exist
            if data.get("resultsCount", 0) == 0 or "results" not in data:
                logger.warning(f"No data returned for {ticker} from {start_str} to {end_str}")
                return pd.DataFrame()
            
            # Process results into DataFrame
            results = data["results"]
            
            # Create DataFrame
            df = pd.DataFrame(results)
            
            # Rename columns to match OHLCV format
            df.rename(columns={
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "t": "timestamp"
            }, inplace=True)
            
            # Convert timestamp from milliseconds to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            
            # Select only OHLCV columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Save data to CSV for future use
            self.csv_provider.save_data(df, instrument, timeframe)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from Polygon: {str(e)}")
            return pd.DataFrame()
    
    def get_instruments(self) -> List[str]:
        """Get list of available instruments"""
        # For Polygon, this would require additional implementation
        # For now, just return what's available in the CSV provider
        return self.csv_provider.get_instruments()
    
    def get_latest_data(self, instrument: str, count: int = 1, timeframe: str = "day") -> pd.DataFrame:
        """Get the latest N data points for an instrument"""
        # For real-time use, would implement a call to Polygon's latest data endpoint
        # For backtesting purposes, can just get the last N rows of historical data
        
        # Get current date/time
        end_date = datetime.datetime.now()
        
        # Calculate start date based on timeframe and count
        if timeframe == 'minute':
            start_date = end_date - datetime.timedelta(minutes=count*2)
        elif timeframe == 'hour':
            start_date = end_date - datetime.timedelta(hours=count*2)
        elif timeframe == 'day':
            start_date = end_date - datetime.timedelta(days=count*2)
        elif timeframe == 'week':
            start_date = end_date - datetime.timedelta(weeks=count*2)
        elif timeframe == 'month':
            start_date = end_date - datetime.timedelta(days=count*30*2)
        else:
            start_date = end_date - datetime.timedelta(days=count*2)
        
        # Get historical data for the period
        data = self.get_historical_data(instrument, start_date, end_date, timeframe)
        
        # Return last N rows
        return data.tail(count)


# ======== Utility Functions ========

def load_data_from_csv(file_path: str) -> pd.DataFrame:
    """
    Load OHLCV data from a CSV file
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        DataFrame with OHLCV data
    """
    try:
        df = pd.read_csv(file_path)
        
        # Check for required columns
        required_columns = ['datetime', 'open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"CSV file missing required columns: {required_columns}")
            return pd.DataFrame()
        
        # Parse datetime column
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        
        return df
    except Exception as e:
        logger.error(f"Error loading data from CSV: {str(e)}")
        return pd.DataFrame()

def download_forex_data(symbol: str, start_date: str, end_date: str, 
                       timeframe: str = "day", save_path: str = None) -> pd.DataFrame:
    """
    Download forex data using Alpha Vantage API (free alternative to Polygon)
    
    Args:
        symbol: Currency pair (e.g., "EUR/USD")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        timeframe: Data timeframe (daily, weekly, monthly)
        save_path: Path to save CSV file
        
    Returns:
        DataFrame with OHLCV data
    """
    # Alpha Vantage API key (free, but limited to 25 calls/day)
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    
    if not api_key:
        logger.error("Alpha Vantage API key not found in environment variables")
        return pd.DataFrame()
    
    # Format symbol for Alpha Vantage
    if '/' in symbol:
        from_currency, to_currency = symbol.split('/')
    else:
        if len(symbol) == 6:
            from_currency = symbol[:3]
            to_currency = symbol[3:]
        else:
            logger.error(f"Invalid forex symbol format: {symbol}")
            return pd.DataFrame()
    
    # Map timeframe to Alpha Vantage interval
    interval_map = {
        "day": "daily",
        "week": "weekly",
        "month": "monthly"
    }
    
    interval = interval_map.get(timeframe, "daily")
    
    # Build URL
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": f"FX_{interval.upper()}",
        "from_symbol": from_currency,
        "to_symbol": to_currency,
        "outputsize": "full",
        "apikey": api_key
    }
    
    try:
        # Make API request
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for error message
        if "Error Message" in data:
            logger.error(f"Alpha Vantage API error: {data['Error Message']}")
            return pd.DataFrame()
        
        # Extract time series data
        time_series_key = f"Time Series FX ({interval.title()})"
        if time_series_key not in data:
            logger.error(f"No time series data found in response. Keys: {data.keys()}")
            return pd.DataFrame()
        
        time_series = data[time_series_key]
        
        # Process data into DataFrame
        records = []
        for date, values in time_series.items():
            records.append({
                'datetime': date,
                'open': float(values['1. open']),
                'high': float(values['2. high']),
                'low': float(values['3. low']),
                'close': float(values['4. close']),
                'volume': 0  # Alpha Vantage doesn't provide volume for forex
            })
        
        df = pd.DataFrame(records)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Filter by date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)]
        
        # Sort by date
        df.sort_values('datetime', inplace=True)
        df.set_index('datetime', inplace=True)
        
        # Save to CSV if path provided
        if save_path:
            df.to_csv(save_path)
            logger.info(f"Data saved to {save_path}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error downloading forex data: {str(e)}")
        return pd.DataFrame()

# ======== Main Function ========

def main():
    """Main function to demonstrate backtesting engine usage"""
    # Create a data provider
    data_provider = CSVDataProvider()
    
    # Create backtester
    backtester = Backtester(data_provider)
    
    # Create a strategy
    strategy = SMA_Crossover_Strategy(fast_period=10, slow_period=30)
    
    # Run backtest
    result = backtester.run_backtest(
        strategy=strategy,
        instrument="EUR/USD",
        start_date="2020-01-01",
        end_date="2021-01-01",
        timeframe="day",
        initial_balance=10000.0,
        commission=0.0001,
        slippage=0.0001,
        risk_per_trade=0.02,
        enable_stop_loss=True,
        enable_take_profit=True
    )
    
    if result:
        # Generate report
        backtester.generate_report(result)
        
        # Plot equity curve
        result.plot_equity_curve()
        
        # Save result to file
        result.save_to_file()
    
    # Demonstrate parameter optimization
    param_ranges = {
        'fast_period': [5, 10, 15, 20],
        'slow_period': [20, 30, 40, 50]
    }
    
    optimization_results = backtester.run_parameter_optimization(
        strategy_class=SMA_Crossover_Strategy,
        parameter_ranges=param_ranges,
        instrument="EUR/USD",
        start_date="2020-01-01",
        end_date="2021-01-01",
        timeframe="day",
        optimization_metric="sharpe_ratio"
    )
    
    # Demonstrate walk-forward analysis
    walk_forward_result = backtester.run_walk_forward_analysis(
        strategy_class=SMA_Crossover_Strategy,
        parameter_ranges=param_ranges,
        instrument="EUR/USD",
        start_date="2020-01-01",
        end_date="2021-01-01",
        timeframe="day",
        train_size=180,
        test_size=90
    )
    
    if walk_forward_result:
        # Generate report for walk-forward results
        backtester.generate_report(walk_forward_result, include_monte_carlo=True)

if __name__ == "__main__":
    main()