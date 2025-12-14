#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train and test on the same forex pair with time-based split
Usage:
    python train_test_same_pair.py <csv_file> <pair_name> [train_first_date] [train_last_date] [test_first_date] [test_last_date]
    python train_test_same_pair.py <csv_file> <pair_name> --one-year
    python train_test_same_pair.py <csv_file> <pair_name> --train-2years-test-1year
    python train_test_same_pair.py <csv_file> <pair_name> --2y-1y
    
Example:
    python train_test_same_pair.py EURUSD_M15.csv "EUR/USD"
    python train_test_same_pair.py EURUSD_M15.csv "EUR/USD" "2023-01-01 00:00:00" "2023-12-31 23:45:00" "2024-01-01 00:00:00" "2024-06-30 23:45:00"
    python train_test_same_pair.py EURUSD_M15.csv "EUR/USD" --one-year
    python train_test_same_pair.py EURUSD_M15.csv "EUR/USD" --train-2years-test-1year
"""

import os
import sys
import subprocess
import shutil
import pandas as pd

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Error: {description} failed with exit code {result.returncode}")
        return False
    return True

def load_csv_data(csv_file):
    """Load CSV data with various delimiter/format attempts"""
    try:
        try:
            data = pd.read_csv(csv_file, delimiter='\t', index_col=0, parse_dates=True, 
                              names=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'], header=0)
        except:
            try:
                data = pd.read_csv(csv_file, delimiter=',', index_col='Time', parse_dates=True)
            except:
                data = pd.read_csv(csv_file, delimiter=',', index_col=0, parse_dates=True,
                                 names=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'], header=0)
        
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        return data
    except Exception as e:
        raise ValueError(f"Could not load CSV: {e}")

def check_date_range(csv_file, first_date, last_date):
    """Check if date range is valid for the CSV file"""
    try:
        data = load_csv_data(csv_file)
        
        print(f"\nData range in CSV: {data.index[0]} to {data.index[-1]}")
        print(f"Total data points: {len(data)}")
        
        if first_date:
            first_dt = pd.to_datetime(first_date)
            if first_dt < data.index[0] or first_dt > data.index[-1]:
                print(f"Warning: Start date {first_date} is outside data range")
        if last_date:
            last_dt = pd.to_datetime(last_date)
            if last_dt < data.index[0] or last_dt > data.index[-1]:
                print(f"Warning: End date {last_date} is outside data range")
        
        return True
    except Exception as e:
        print(f"Warning: Could not check date range: {e}")
        return False

def suggest_one_year_periods(csv_file):
    """Suggest 1-year training and testing periods from available data"""
    try:
        data = load_csv_data(csv_file)
        start_date = data.index[0]
        end_date = data.index[-1]
        
        # Calculate total time span
        time_span = end_date - start_date
        years_available = time_span.days / 365.25
        
        print(f"\nAvailable data: {start_date} to {end_date}")
        print(f"Time span: {years_available:.2f} years")
        
        if years_available >= 2:
            # Suggest first year for training, second year for testing
            train_start = start_date
            train_end = start_date + pd.DateOffset(years=1)
            test_start = train_end
            test_end = min(test_start + pd.DateOffset(years=1), end_date)
            
            print(f"\nSuggested 1-year periods:")
            print(f"  Training: {train_start} to {train_end}")
            print(f"  Testing:  {test_start} to {test_end}")
            return train_start, train_end, test_start, test_end
        else:
            print(f"\nWarning: Less than 2 years of data available. Cannot split into 1-year train/test periods.")
            return None, None, None, None
    except Exception as e:
        print(f"Warning: Could not suggest periods: {e}")
        return None, None, None, None

def suggest_two_year_train_one_year_test_periods(csv_file):
    """Suggest 2-year training and 1-year testing periods from available data"""
    try:
        data = load_csv_data(csv_file)
        start_date = data.index[0]
        end_date = data.index[-1]
        
        # Calculate total time span
        time_span = end_date - start_date
        years_available = time_span.days / 365.25
        
        print(f"\nAvailable data: {start_date} to {end_date}")
        print(f"Time span: {years_available:.2f} years")
        
        if years_available >= 3:
            # Suggest first 2 years for training, next 1 year for testing
            train_start = start_date
            train_end = start_date + pd.DateOffset(years=2)
            test_start = train_end
            test_end = min(test_start + pd.DateOffset(years=1), end_date)
            
            print(f"\nSuggested 2-year training / 1-year test periods:")
            print(f"  Training: {train_start} to {train_end} (2 years)")
            print(f"  Testing:  {test_start} to {test_end} (1 year)")
            return train_start, train_end, test_start, test_end
        else:
            print(f"\nWarning: Less than 3 years of data available. Cannot split into 2-year train / 1-year test periods.")
            print(f"  Available: {years_available:.2f} years, Required: 3 years")
            return None, None, None, None
    except Exception as e:
        print(f"Warning: Could not suggest periods: {e}")
        return None, None, None, None

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nError: Missing required arguments")
        print("Required: csv_file pair_name")
        print("Optional: train_first_date train_last_date test_first_date test_last_date")
        print("\nFor automatic period suggestions, use:")
        print("  python train_test_same_pair.py <csv_file> <pair_name> --one-year")
        print("  python train_test_same_pair.py <csv_file> <pair_name> --train-2years-test-1year")
        print("  python train_test_same_pair.py <csv_file> <pair_name> --2y-1y")
        print("  or specify dates manually")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    pair_name = sys.argv[2]
    
    # Check for flags
    use_one_year = '--one-year' in sys.argv
    use_two_year_train = '--train-2years-test-1year' in sys.argv or '--2y-1y' in sys.argv
    
    # Filter out flags from arguments
    args_without_flags = [arg for arg in sys.argv[3:] if arg not in ['--one-year', '--train-2years-test-1year', '--2y-1y']]
    
    # Optional date ranges
    train_first_date = args_without_flags[0] if len(args_without_flags) > 0 else None
    train_last_date = args_without_flags[1] if len(args_without_flags) > 1 else None
    test_first_date = args_without_flags[2] if len(args_without_flags) > 2 else None
    test_last_date = args_without_flags[3] if len(args_without_flags) > 3 else None
    
    # Ensure CSV file is in data-cache
    if not os.path.isabs(csv_file) and '/' not in csv_file and '\\' not in csv_file:
        csv_file_path = os.path.join('data-cache', csv_file)
    else:
        csv_file_path = csv_file
    
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found: {csv_file_path}")
        sys.exit(1)
    
    # If flags are set, suggest periods
    if use_two_year_train and not (train_first_date and train_last_date and test_first_date and test_last_date):
        train_start, train_end, test_start, test_end = suggest_two_year_train_one_year_test_periods(csv_file_path)
        if train_start is not None:
            train_first_date = str(train_start)
            train_last_date = str(train_end)
            test_first_date = str(test_start)
            test_last_date = str(test_end)
            print(f"\nUsing suggested 2-year training / 1-year test periods:")
            print(f"  Training: {train_first_date} to {train_last_date} (2 years)")
            print(f"  Testing:  {test_first_date} to {test_last_date} (1 year)")
        else:
            print("\nError: Cannot determine 2-year train / 1-year test periods. Please specify dates manually.")
            sys.exit(1)
    elif use_one_year and not (train_first_date and train_last_date and test_first_date and test_last_date):
        train_start, train_end, test_start, test_end = suggest_one_year_periods(csv_file_path)
        if train_start is not None:
            train_first_date = str(train_start)
            train_last_date = str(train_end)
            test_first_date = str(test_start)
            test_last_date = str(test_end)
            print(f"\nUsing suggested 1-year periods:")
            print(f"  Training: {train_first_date} to {train_last_date}")
            print(f"  Testing:  {test_first_date} to {test_last_date}")
        else:
            print("\nError: Cannot determine 1-year periods. Please specify dates manually.")
            sys.exit(1)
    
    # Create unique folder names based on pair name
    pair_clean = pair_name.lower().replace('/', '_')
    
    train_images_dir = f"chart_images_{pair_clean}_train"
    train_split_dir = f"{pair_clean}_splitted_w5_s2"
    model_name = f"chart_classification_model_{pair_clean}.h5"
    
    print(f"\n{'='*60}")
    print("TRAIN-TEST SAME PAIR WORKFLOW")
    print(f"{'='*60}")
    print(f"Pair: {pair_name} ({csv_file})")
    print(f"Training images directory: {train_images_dir}")
    print(f"Training split directory: {train_split_dir}")
    print(f"Model name: {model_name}")
    if train_first_date or train_last_date:
        print(f"Training date range: {train_first_date or 'start'} to {train_last_date or 'end'}")
    if test_first_date or test_last_date:
        print(f"Testing date range: {test_first_date or 'start'} to {test_last_date or 'end'}")
    print(f"{'='*60}\n")
    
    # Check date ranges if provided
    if train_first_date or train_last_date or test_first_date or test_last_date:
        check_date_range(csv_file_path, train_first_date or test_first_date, train_last_date or test_last_date)
    
    # Step 1: Generate training images
    print("\nStep 1: Generating training images...")
    if os.path.exists(train_images_dir):
        # Check if directory is empty or if we should regenerate
        existing_files = [f for f in os.listdir(train_images_dir) if os.path.isfile(os.path.join(train_images_dir, f))]
        if len(existing_files) == 0:
            print(f"Directory {train_images_dir} exists but is empty. Regenerating...")
            shutil.rmtree(train_images_dir, ignore_errors=True)
        else:
            # For non-interactive mode, use existing if directory has files
            try:
                response = input(f"Directory {train_images_dir} already exists. Delete and regenerate? (y/n): ")
                if response.lower() == 'y':
                    shutil.rmtree(train_images_dir, ignore_errors=True)
                else:
                    print(f"Using existing images in {train_images_dir}")
            except EOFError:
                # Non-interactive mode - use existing if it has content
                print(f"Using existing images in {train_images_dir} (non-interactive mode)")
    
    if not os.path.exists(train_images_dir):
        cmd = [sys.executable, "Pattern_Technical indicators _Next Close price.py", csv_file_path, train_images_dir]
        # Add date range if specified
        if train_first_date:
            cmd.append(train_first_date)
        if train_last_date:
            cmd.append(train_last_date)
        if not run_command(cmd, "Generating training images"):
            sys.exit(1)
    
    # Step 2: Train the model
    print("\nStep 2: Training CNN model...")
    if os.path.exists(model_name):
        try:
            response = input(f"Model {model_name} already exists. Retrain? (y/n): ")
            if response.lower() != 'y':
                print(f"Using existing model: {model_name}")
            else:
                cmd = [sys.executable, "CNN.py", train_images_dir, train_split_dir, model_name]
                if not run_command(cmd, "Training CNN model"):
                    sys.exit(1)
        except EOFError:
            # Non-interactive mode - retrain if model exists
            print(f"Model {model_name} exists. Retraining...")
            cmd = [sys.executable, "CNN.py", train_images_dir, train_split_dir, model_name]
            if not run_command(cmd, "Training CNN model"):
                sys.exit(1)
    else:
        cmd = [sys.executable, "CNN.py", train_images_dir, train_split_dir, model_name]
        if not run_command(cmd, "Training CNN model"):
            sys.exit(1)
    
    # Step 3: Test on same pair (different time period if specified)
    print("\nStep 3: Testing on same pair...")
    # Temporarily rename the model so simulate_forex_pair.py can use it
    temp_model_name = "chart_classification_model.h5"
    if os.path.exists(temp_model_name):
        backup_name = temp_model_name + ".backup"
        if os.path.exists(backup_name):
            os.remove(backup_name)
        os.rename(temp_model_name, backup_name)
    
    # Copy the trained model to the default name
    shutil.copy(model_name, temp_model_name)
    
    try:
        cmd = [sys.executable, "simulate_forex_pair.py", csv_file_path, pair_name]
        if test_first_date:
            cmd.append(test_first_date)
        if test_last_date:
            cmd.append(test_last_date)
        
        if not run_command(cmd, f"Testing on {pair_name}"):
            sys.exit(1)
    finally:
        # Restore original model if it existed
        if os.path.exists(backup_name):
            os.remove(temp_model_name)
            os.rename(backup_name, temp_model_name)
        elif os.path.exists(temp_model_name):
            os.remove(temp_model_name)
    
    # Step 4: Calculate metrics
    result_file = f'trade_result_{pair_clean}.txt'
    if os.path.exists(result_file):
        print("\nStep 4: Calculating performance metrics...")
        cmd = [sys.executable, "calculate_metrics.py", result_file]
        run_command(cmd, "Calculating metrics")
    
    print(f"\n{'='*60}")
    print("WORKFLOW COMPLETE!")
    print(f"{'='*60}")
    print(f"Trained model: {model_name}")
    print(f"Test results: {result_file}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

