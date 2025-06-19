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


def setup_argparse():
    """
    Setup command line argument parser
    """
    parser = argparse.ArgumentParser(
        description="Nifty Options Algo - A modular options trading algorithm"
    )
    
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "-m", "--mode", 
        type=str, 
        choices=["live", "simulation"], 
        default="simulation",
        help="Trading mode: live or simulation (default: simulation)"
    )
    
    parser.add_argument(
        "-b", "--backtest", 
        action="store_true", 
        help="Run in backtest mode"
    )
    
    parser.add_argument(
        "--start-date", 
        type=str, 
        help="Start date for backtest (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end-date", 
        type=str, 
        help="End date for backtest (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--status", 
        action="store_true", 
        help="Print strategy status and exit"
    )
    
    return parser


def validate_args(args):
    """
    Validate command line arguments
    """
    # Check if config file exists
    if args.config and not os.path.isfile(args.config):
        print(f"Error: Config file '{args.config}' not found")
        return False
    
    # Check backtest arguments
    if args.backtest:
        if not args.start_date or not args.end_date:
            print("Error: Both --start-date and --end-date are required for backtest mode")
            return False
        
        try:
            datetime.strptime(args.start_date, "%Y-%m-%d")
            datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            print("Error: Date format should be YYYY-MM-DD")
            return False
    
    return True


def main():
    """
    Main entry point for the application
    """
    # Parse command line arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Validate arguments
    if not validate_args(args):
        sys.exit(1)
    
    try:
        # Initialize the strategy
        strategy = NiftyOptionsStrategy(args.config)
        
        # Override mode if specified
        if args.mode:
            strategy.config['mode'] = args.mode
        
        # Print status and exit if requested
        if args.status:
            status = strategy.get_status()
            print(json.dumps(status, indent=2))
            sys.exit(0)
        
        # Run backtest if requested
        if args.backtest:
            results = strategy.backtest(args.start_date, args.end_date)
            print(json.dumps(results, indent=2))
            sys.exit(0)
        
        # Start the strategy
        print(f"Starting Nifty Options Algo in {strategy.config['mode']} mode...")
        strategy.start()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        logging.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
