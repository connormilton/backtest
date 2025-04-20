#!/usr/bin/env python3
"""
Strategy Discovery Bot
Generates, tests, and ranks trading strategies using the backtesting engine
"""

import os
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Union

# Configure logging
logger = logging.getLogger("Strategy_Discovery")

class StrategyDiscoveryBot:
    """
    Bot that generates and tests trading strategies using the backtesting engine
    """
    
    def __init__(self, backtester, llm_client=None):
        """
        Initialize strategy discovery bot
        
        Args:
            backtester: Backtesting engine instance
            llm_client: LLM client for strategy generation (optional)
        """
        self.backtester = backtester
        self.llm_client = llm_client
        self.strategy_library = "strategies/"
        os.makedirs(self.strategy_library, exist_ok=True)
    
    def discover_strategies(self, num_iterations=10, instrument="EUR/USD", 
                           timeframe="H1", eval_period=30):
        """
        Discover new trading strategies through generation and testing
        
        Args:
            num_iterations: Number of strategies to generate and test
            instrument: Instrument to test on
            timeframe: Timeframe for testing
            eval_period: Days to use for evaluation
            
        Returns:
            List of successful strategies
        """
        successful_strategies = []
        
        for i in range(num_iterations):
            try:
                logger.info(f"Strategy discovery iteration {i+1}/{num_iterations}")
                
                # Generate strategy concept
                strategy_concept = self._generate_strategy_concept()
                
                # Convert concept to code
                strategy_code = self._generate_strategy_code(strategy_concept)
                
                # Create strategy instance
                strategy = self._create_strategy_from_code(strategy_code)
                
                # Calculate date range
                end_date = datetime.datetime.now()
                start_date = end_date - datetime.timedelta(days=eval_period)
                
                # Test strategy
                result = self.backtester.run_backtest(
                    strategy=strategy,
                    instrument=instrument,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=timeframe,
                    initial_balance=10000.0
                )
                
                # Evaluate and save if successful
                if self._is_successful(result):
                    strategy_info = self._save_strategy(strategy, result, strategy_code)
                    successful_strategies.append(strategy_info)
                    logger.info(f"Discovered successful strategy: {strategy.name}")
                
            except Exception as e:
                logger.error(f"Error in strategy discovery iteration: {e}")
        
        return successful_strategies
    
    def _generate_strategy_concept(self):
        """Generate a trading strategy concept"""
        if self.llm_client:
            # Use LLM to generate concept
            prompt = """
            Generate a forex trading strategy concept for EUR/USD. 
            Include:
            1. Strategy name
            2. Core logic (indicators, signals)
            3. Entry and exit rules
            4. Risk management approach
            """
            return self.llm_client.generate(prompt)
        else:
            # Return a basic strategy concept template
            return {
                "name": f"Strategy_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "type": "technical",
                "indicators": ["moving_average", "rsi"],
                "logic": "Buy when RSI < 30 and fast MA crosses above slow MA"
            }
    
    def _generate_strategy_code(self, strategy_concept):
        """Generate code from strategy concept"""
        if self.llm_client:
            # Use LLM to generate code
            prompt = f"""
            Create a trading strategy class in Python that inherits from Strategy base class.
            
            Strategy concept: {strategy_concept}
            
            The class should implement the generate_signals method that accepts a DataFrame
            with OHLCV data and returns the DataFrame with a 'signal' column added.
            Signal values: 1 (buy), -1 (sell), 0 (no position)
            
            Also implement calculate_stop_loss and calculate_take_profit methods.
            """
            return self.llm_client.generate(prompt)
        else:
            # Return a basic strategy code template
            return """
class DiscoveredStrategy(Strategy):
    def __init__(self, name="Discovered Strategy", fast_period=10, slow_period=30, rsi_period=14, rsi_threshold=30):
        super().__init__(name)
        self.parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'rsi_period': rsi_period,
            'rsi_threshold': rsi_threshold
        }
    
    def generate_signals(self, data):
        if data.empty:
            return data
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Calculate indicators
        fast_period = self.parameters['fast_period']
        slow_period = self.parameters['slow_period']
        rsi_period = self.parameters['rsi_period']
        rsi_threshold = self.parameters['rsi_threshold']
        
        # Calculate moving averages
        df['fast_ma'] = df['close'].rolling(window=fast_period).mean()
        df['slow_ma'] = df['close'].rolling(window=slow_period).mean()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Initialize signal column
        df['signal'] = 0
        
        # Generate signals
        df['prev_fast_ma'] = df['fast_ma'].shift(1)
        df['prev_slow_ma'] = df['slow_ma'].shift(1)
        
        # Buy signal: RSI oversold and fast MA crosses above slow MA
        buy_condition = (df['rsi'] < rsi_threshold) & \
                        (df['prev_fast_ma'] < df['prev_slow_ma']) & \
                        (df['fast_ma'] > df['slow_ma'])
        df.loc[buy_condition, 'signal'] = 1
        
        # Sell signal: fast MA crosses below slow MA
        sell_condition = (df['prev_fast_ma'] > df['prev_slow_ma']) & \
                         (df['fast_ma'] < df['slow_ma'])
        df.loc[sell_condition, 'signal'] = -1
        
        return df
    
    def calculate_stop_loss(self, data, signal_row_idx, direction):
        # Look back for recent swing points
        lookback = min(20, signal_row_idx)
        recent_data = data.iloc[signal_row_idx-lookback:signal_row_idx+1]
        
        if direction == "BUY":
            # For buy signals, set stop below recent low
            stop_price = recent_data['low'].min() * 0.995  # Slightly below the low
        else:
            # For sell signals, set stop above recent high
            stop_price = recent_data['high'].max() * 1.005  # Slightly above the high
            
        return stop_price
    
    def calculate_take_profit(self, data, signal_row_idx, direction):
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
"""
    
    def _create_strategy_from_code(self, strategy_code):
        """Create strategy instance from code"""
        try:
            # Create a namespace to execute the code
            namespace = {}
            
            # Add required imports and classes to namespace
            exec("from engine import Strategy", namespace)
            
            # Execute the strategy code
            exec(strategy_code, namespace)
            
            # Find the strategy class in the namespace
            strategy_classes = [cls for name, cls in namespace.items() 
                              if isinstance(cls, type) and issubclass(cls, namespace["Strategy"]) 
                              and cls != namespace["Strategy"]]
            
            if not strategy_classes:
                raise ValueError("No Strategy subclass defined in the code")
            
            # Create an instance of the first strategy class found
            strategy_class = strategy_classes[0]
            strategy_instance = strategy_class()
            
            # Store the code in the strategy for saving
            strategy_instance.code = strategy_code
            
            return strategy_instance
            
        except Exception as e:
            logger.error(f"Error creating strategy from code: {e}")
            raise
    
    def _is_successful(self, result):
        """Determine if a strategy is worth keeping"""
        if result is None:
            return False
            
        # Minimum criteria for a successful strategy
        return (result.sharpe_ratio > 1.0 and
                result.win_rate > 50 and
                result.profit_factor > 1.2 and
                result.max_drawdown_pct < 25)
    
    def _save_strategy(self, strategy, result, strategy_code):
        """Save a successful strategy with its performance metrics"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_name = f"{strategy.name}_{timestamp}"
        
        # Create strategy folder
        strategy_dir = os.path.join(self.strategy_library, strategy_name)
        os.makedirs(strategy_dir, exist_ok=True)
        
        # Save strategy code
        with open(os.path.join(strategy_dir, "strategy.py"), "w") as f:
            f.write(strategy_code)
        
        # Save performance metrics
        with open(os.path.join(strategy_dir, "performance.json"), "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        # Create a summary file
        summary = {
            "name": strategy.name,
            "timestamp": timestamp,
            "parameters": strategy.parameters,
            "performance": {
                "sharpe_ratio": result.sharpe_ratio,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
                "max_drawdown_pct": result.max_drawdown_pct,
                "total_return_pct": result.total_return_pct
            }
        }
        
        with open(os.path.join(strategy_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def list_successful_strategies(self, sort_by="sharpe_ratio"):
        """
        List all successful strategies in the library
        
        Args:
            sort_by: Metric to sort by (sharpe_ratio, win_rate, profit_factor, total_return_pct)
            
        Returns:
            List of strategy summaries
        """
        strategies = []
        
        # Iterate through strategy directories
        if os.path.exists(self.strategy_library):
            for strategy_dir in os.listdir(self.strategy_library):
                summary_file = os.path.join(self.strategy_library, strategy_dir, "summary.json")
                
                if os.path.exists(summary_file):
                    try:
                        with open(summary_file, 'r') as f:
                            summary = json.load(f)
                            
                        # Add strategy to list
                        strategies.append(summary)
                    except:
                        continue
        
        # Sort strategies
        if strategies:
            try:
                strategies.sort(
                    key=lambda x: x.get("performance", {}).get(sort_by, 0), 
                    reverse=True
                )
            except:
                # If sorting fails, don't sort
                pass
                
        return strategies
    
    def load_strategy(self, strategy_name):
        """
        Load a strategy from the library
        
        Args:
            strategy_name: Name of the strategy to load
            
        Returns:
            Strategy instance
        """
        strategy_dir = os.path.join(self.strategy_library, strategy_name)
        strategy_file = os.path.join(strategy_dir, "strategy.py")
        
        if not os.path.exists(strategy_file):
            raise ValueError(f"Strategy file not found: {strategy_file}")
            
        # Read strategy code
        with open(strategy_file, 'r') as f:
            strategy_code = f.read()
            
        # Create strategy instance
        return self._create_strategy_from_code(strategy_code)
    
    def generate_random_parameters(self, strategy_name, num_variations=5):
        """
        Generate random parameter variations for a strategy
        
        Args:
            strategy_name: Name of the strategy to vary
            num_variations: Number of parameter variations to generate
            
        Returns:
            List of parameter sets
        """
        import random
        
        # Load the strategy
        strategy = self.load_strategy(strategy_name)
        base_params = strategy.parameters
        
        # Define parameter ranges based on current values
        param_ranges = {}
        for param, value in base_params.items():
            if isinstance(value, int):
                param_ranges[param] = (max(1, int(value * 0.5)), int(value * 1.5))
            elif isinstance(value, float):
                param_ranges[param] = (value * 0.5, value * 1.5)
            else:
                # For non-numeric parameters, don't vary
                param_ranges[param] = (value, value)
                
        # Generate variations
        variations = []
        for _ in range(num_variations):
            new_params = {}
            for param, (min_val, max_val) in param_ranges.items():
                if isinstance(base_params[param], int):
                    new_params[param] = random.randint(min_val, max_val)
                elif isinstance(base_params[param], float):
                    new_params[param] = random.uniform(min_val, max_val)
                else:
                    new_params[param] = base_params[param]
            
            variations.append(new_params)
            
        return variations
    
    def evolve_strategies(self, num_generations=3, population_size=5, instrument="EUR/USD"):
        """
        Evolve strategies through multiple generations
        
        Args:
            num_generations: Number of generations to evolve
            population_size: Size of the population in each generation
            instrument: Instrument to test on
            
        Returns:
            List of evolved strategies
        """
        # Start with existing successful strategies
        strategies = self.list_successful_strategies()
        
        if not strategies:
            logger.warning("No existing strategies found, generating initial population")
            # Generate initial population
            self.discover_strategies(num_iterations=population_size, instrument=instrument)
            strategies = self.list_successful_strategies()
            
            if not strategies:
                logger.error("Failed to generate initial population")
                return []
        
        # Evolve through generations
        for generation in range(num_generations):
            logger.info(f"Starting evolution generation {generation+1}/{num_generations}")
            
            # Select top strategies
            top_strategies = strategies[:min(len(strategies), population_size)]
            new_strategies = []
            
            # Generate variations for each top strategy
            for strategy_summary in top_strategies:
                strategy_name = strategy_summary.get("name")
                if not strategy_name:
                    continue
                    
                # Generate parameter variations
                variations = self.generate_random_parameters(strategy_name, num_variations=3)
                
                # Test each variation
                for params in variations:
                    try:
                        # Load strategy
                        strategy = self.load_strategy(strategy_name)
                        
                        # Apply new parameters
                        strategy.set_parameters(**params)
                        
                        # Test strategy
                        end_date = datetime.datetime.now()
                        start_date = end_date - datetime.timedelta(days=30)
                        
                        result = self.backtester.run_backtest(
                            strategy=strategy,
                            instrument=instrument,
                            start_date=start_date,
                            end_date=end_date,
                            timeframe="H1",
                            initial_balance=10000.0
                        )
                        
                        # Save if successful
                        if self._is_successful(result):
                            strategy_info = self._save_strategy(strategy, result, strategy.code)
                            new_strategies.append(strategy_info)
                            logger.info(f"Evolved successful strategy: {strategy.name}")
                    except Exception as e:
                        logger.error(f"Error evolving strategy: {e}")
            
            # Update strategies list
            if new_strategies:
                all_strategies = self.list_successful_strategies()
                strategies = all_strategies
                logger.info(f"Generation {generation+1} complete: {len(new_strategies)} new successful strategies")
            else:
                logger.warning(f"No successful strategies in generation {generation+1}")
        
        return self.list_successful_strategies()

# Example usage
if __name__ == "__main__":
    print("Strategy Discovery Bot - Run with an existing backtester instance")