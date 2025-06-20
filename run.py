import os
import sys
import argparse
import logging
import json
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the strategy
from core.strategy import NiftyOptionsStrategy


def setup_argparse() -> argparse.ArgumentParser:
    """
    Setup command line argument parser

    Returns:
        argparse.ArgumentParser: Command line argument parser
    """
    # Create an ArgumentParser object with a description for the script
    parser = argparse.ArgumentParser(
        description="Nifty Options Algo - A modular options trading algorithm"
    )
    
    # Add an argument for the configuration file path
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        default="config/default_config.ini",  # Default path to the config file
        help="Path to configuration file"
    )
    
    # Add an argument to specify the run mode, either 'simulation' or 'live'
    parser.add_argument(
        "--mode",
        type=str,
        choices=["live"],  # Limit choices to live
        default="live",  # Default to live mode
        help="Run mode (live)"
    )
    
    # Add a flag to enable backtest mode
    parser.add_argument(
        "--backtest",
        action="store_true",  # Store True if this flag is used
        help="Run in backtest mode"
    )
    
    # Add an argument for the backtest start date
    parser.add_argument(
        "--start_date",
        type=str,  # Expect a string in YYYY-MM-DD format
        help="Backtest start date (YYYY-MM-DD)"
    )
    
    # Add an argument for the backtest end date
    parser.add_argument(
        "--end_date",
        type=str,  # Expect a string in YYYY-MM-DD format
        help="Backtest end date (YYYY-MM-DD)"
    )
    
    # Add a flag to print the strategy status and exit
    parser.add_argument(
        "--status", 
        action="store_true",  # Store True if this flag is used
        help="Print strategy status and exit"
    )
    
    return parser


def validate_args(args) -> bool:
    """
    Validate command line arguments
    
    Checks:
    1. If the config file specified by --config exists
    2. If --backtest is specified, checks that both --start_date and --end_date are provided
    3. If --backtest is specified, checks that the start and end dates are in YYYY-MM-DD format
    
    Returns:
    True if all arguments are valid, False otherwise 
    """
    # 1. Check if the config file specified by --config exists
    if args.config:
        if not os.path.isfile(args.config):
            print(f"Error: Config file '{args.config}' not found")
            return False
    
    # 2. Check backtest arguments
    if args.backtest:
        # Check that both --start_date and --end_date are provided
        if not args.start_date or not args.end_date:
            print("Error: Both --start_date and --end_date are required for backtest mode")
            return False
        
        # 3. Check that the start and end dates are in YYYY-MM-DD format
        try:
            # Attempt to parse the dates with the format YYYY-MM-DD
            datetime.strptime(args.start_date, "%Y-%m-%d")
            datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            # If parsing fails, print an error message and return False
            print("Error: Date format should be YYYY-MM-DD")
            return False
    
    # If all checks pass, return True
    return True


def main():
    """
    Main entry point for the application.
    
    This function orchestrates the initialization and execution of the Nifty Options
    Algo trading strategy. It handles command line argument parsing, validation, and
    determines the execution path based on user inputs. The strategy can be executed
    in live mode, simulation mode, or backtest mode. It also provides an option to
    simply print the current strategy status and exit.
    
    Command Line Arguments:
    - `--config`: Path to the configuration file (default: "config/default_config.ini").
    - `--mode`: Run mode, either 'simulation' or 'live' (default: 'live').
    - `--backtest`: Flag to run in backtest mode.
    - `--start_date`: Start date for backtest in 'YYYY-MM-DD' format.
    - `--end_date`: End date for backtest in 'YYYY-MM-DD' format.
    - `--status`: Flag to print the strategy status and exit.
    
    Exits:
    - Exits with code 0 on successful completion or user interruption.
    - Exits with code 1 on argument validation failure or other exceptions.
    """
    # Parse command line arguments using the argument parser
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Validate the parsed command line arguments
    if not validate_args(args):
        sys.exit(1)  # Exit with code 1 if validation fails
    
    try:
        # Initialize the trading strategy with the specified configuration and mode
        strategy = NiftyOptionsStrategy(config_path=args.config, mode=args.mode)
        
        # If the --status flag is provided, print the current strategy status and exit
        if args.status:
            status = strategy.get_status()
            print(json.dumps(status, indent=2))  # Print the status as a formatted JSON
            sys.exit(0)  # Exit with code 0 after printing the status
        
        # If the --backtest flag is provided, run the backtest with the specified dates
        if args.backtest:
            results = strategy.backtest(args.start_date, args.end_date)
            print(json.dumps(results, indent=2))  # Print the backtest results as formatted JSON
            sys.exit(0)  # Exit with code 0 after printing the backtest results
        
        # If neither --status nor --backtest is specified, start the strategy in the given mode
        print(f"Starting Nifty Options Algo in {args.mode} mode...")
        strategy.start()  # Start the trading strategy execution
        
    except KeyboardInterrupt:
        # Handle user interruption gracefully
        print("\nInterrupted by user. Exiting...")
        sys.exit(0)  # Exit with code 0 on user interruption
    except Exception as e:
        # Log any unexpected exceptions and exit with an error code
        print(f"Error: {e}")
        logging.error(f"Error: {e}", exc_info=True)  # Log the error with traceback
        sys.exit(1)  # Exit with code 1 on exception


if __name__ == "__main__":
    main()
