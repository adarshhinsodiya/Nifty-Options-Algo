# Nifty Options Algo

A modular, maintainable Python package for algorithmic trading of NIFTY options. This project implements a technical analysis-based options trading strategy with support for both live trading and simulation modes.

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for data handling, signal generation, execution, and strategy orchestration
- **Technical Analysis**: Implements candle pattern recognition and technical indicator analysis for signal generation
- **Position Management**: Comprehensive position tracking with stop-loss, take-profit, and P&L calculations
- **Simulation Mode**: Full simulation capability for testing strategies without real market execution
- **Live Trading**: Integration with ICICI Direct Breeze Connect API for live market data and order execution
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
   Copy the example environment file and update with your credentials:
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file and add your ICICI Direct Breeze Connect API credentials:
   ```
   # Breeze Connect API Credentials
   ICICI_API_KEY=your_api_key_here
   ICICI_API_SECRET=your_api_secret_here
   ICICI_SESSION_TOKEN=your_session_token_here
   ICICI_USER_ID=your_user_id_here  # Optional
   ```

   To get your API credentials:
   1. Log in to your ICICI Direct account
   2. Go to Developer API section
   3. Generate API Key and Secret
   4. Generate a session token using the API key and secret

## Breeze Connect Integration

This project uses ICICI Direct's Breeze Connect API for live trading. The integration includes:

- **Authentication**: Secure API key and session token management
- **Rate Limiting**: Built-in rate limiting to respect API quotas
- **Error Handling**: Comprehensive error handling and retry mechanisms
- **WebSocket Support**: Optional WebSocket connection for real-time data

### API Rate Limits

Breeze Connect has the following rate limits:
- 200 requests per minute for REST API
- 1 request per second for WebSocket API

The application automatically enforces these limits to prevent rate limiting errors.

### Session Management

The Breeze Connect session token is valid for 24 hours. You'll need to generate a new token daily or implement token refresh logic.

### Testing with Breeze Connect

1. **Sandbox Environment**:
   - Use the Breeze Connect sandbox for testing
   - Update the base URL in the configuration to point to the sandbox environment
   - Test with small order quantities first

2. **Live Trading**:
   - Start with paper trading to validate your strategy
   - Monitor the logs for any API errors or issues
   - Gradually increase position sizes as you gain confidence

### Troubleshooting

- **Authentication Errors**: Verify your API key, secret, and session token
- **Rate Limit Errors**: The application should handle these automatically, but check logs if they persist
- **Connection Issues**: Verify your internet connection and API endpoint URLs
- **Session Timeout**: Implement token refresh logic if needed


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

