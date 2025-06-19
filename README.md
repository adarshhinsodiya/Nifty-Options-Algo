# Nifty Options Algo

A modular, maintainable Python package for algorithmic trading of NIFTY options. This project implements a technical analysis-based options trading strategy with support for both live trading and simulation modes.

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for data handling, signal generation, execution, and strategy orchestration
- **Technical Analysis**: Implements candle pattern recognition and technical indicator analysis for signal generation
- **Position Management**: Comprehensive position tracking with stop-loss, take-profit, and P&L calculations
- **Simulation Mode**: Full simulation capability for testing strategies without real market execution
- **Live Trading**: Integration with DhanHQ API for live market data and order execution
- **Configurable**: Extensive configuration options via INI files
- **Logging**: Comprehensive logging with rotation support

## Project Structure

```
Nifty-Options-Algo/
├── core/                  # Core modules
│   ├── data_handler.py    # Market data fetching and processing
│   ├── signal_generation.py # Signal generation from market data
│   ├── execution.py       # Order execution and position management
│   ├── strategy.py        # Main strategy orchestration
│   └── position.py        # Data classes and enums
├── utils/                 # Utility modules
│   ├── config_loader.py   # Configuration loading and validation
│   ├── logger.py          # Logging setup
│   └── rate_limit.py      # API rate limiting
├── config/                # Configuration files
│   └── default_config.ini # Default configuration
├── data/                  # Data storage directory
├── logs/                  # Log files directory
├── run.py                 # Main entry point
└── README.md              # This file
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Nifty-Options-Algo.git
   cd Nifty-Options-Algo
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables for live trading:
   ```
   # Create a .env file in the project root
   DHAN_CLIENT_ID=your_client_id
   DHAN_ACCESS_TOKEN=your_access_token
   ```

## Usage

### Running in Simulation Mode

```bash
python run.py --mode simulation
```

### Running in Live Mode

```bash
python run.py --mode live
```

### Using a Custom Configuration

```bash
python run.py --config path/to/your/config.ini
```

### Getting Strategy Status

```bash
python run.py --status
```

## Configuration

The strategy can be configured using INI files. See `config/default_config.ini` for available options.

Key configuration sections include:

- **mode**: Live or simulation trading mode
- **capital**: Initial capital and risk parameters
- **strategy**: Strategy-specific parameters like lot size, stop-loss, take-profit
- **expiry**: Options expiry selection (weekly/monthly)
- **api**: API-related settings like throttling
- **logging**: Logging configuration
- **data**: Data fetching and caching parameters

## Development

### Adding New Signal Types

Extend the `SignalGenerator` class in `core/signal_generation.py` to add new technical indicators or candle patterns.

### Adding New Execution Methods

Extend the `ExecutionHandler` class in `core/execution.py` to add new order types or execution strategies.

