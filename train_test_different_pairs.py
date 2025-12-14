#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train on one forex pair and test on another
Usage:
    python train_test_different_pairs.py <train_csv> <test_csv> <train_pair_name> <test_pair_name> [train_date_range] [test_date_range]
    
Example:
    python train_test_different_pairs.py EURUSD_M15.csv GBPUSD15.csv "EUR/USD" "GBP/USD"
    python train_test_different_pairs.py EURUSD_M15.csv GBPUSD15.csv "EUR/USD" "GBP/USD" "2023-01-01 00:00:00" "2024-01-01 00:00:00" "2024-01-01 00:00:00" "2024-06-30 23:45:00"
"""

import os
import sys
import subprocess
import shutil

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

def main():
    if len(sys.argv) < 5:
        print(__doc__)
        print("\nError: Missing required arguments")
        print("Required: train_csv test_csv train_pair_name test_pair_name")
        print("Optional: train_first_date train_last_date test_first_date test_last_date")
        sys.exit(1)
    
    train_csv = sys.argv[1]
    test_csv = sys.argv[2]
    train_pair_name = sys.argv[3]
    test_pair_name = sys.argv[4]
    
    # Optional date ranges
    train_first_date = sys.argv[5] if len(sys.argv) > 5 else None
    train_last_date = sys.argv[6] if len(sys.argv) > 6 else None
    test_first_date = sys.argv[7] if len(sys.argv) > 7 else None
    test_last_date = sys.argv[8] if len(sys.argv) > 8 else None
    
    # Ensure CSV files are in data-cache
    if not os.path.isabs(train_csv) and '/' not in train_csv and '\\' not in train_csv:
        train_csv_path = os.path.join('data-cache', train_csv)
    else:
        train_csv_path = train_csv
    
    if not os.path.isabs(test_csv) and '/' not in test_csv and '\\' not in test_csv:
        test_csv_path = os.path.join('data-cache', test_csv)
    else:
        test_csv_path = test_csv
    
    if not os.path.exists(train_csv_path):
        print(f"Error: Training CSV file not found: {train_csv_path}")
        sys.exit(1)
    
    if not os.path.exists(test_csv_path):
        print(f"Error: Test CSV file not found: {test_csv_path}")
        sys.exit(1)
    
    # Create unique folder names based on pair names
    train_pair_clean = train_pair_name.lower().replace('/', '_')
    test_pair_clean = test_pair_name.lower().replace('/', '_')
    
    train_images_dir = f"chart_images_{train_pair_clean}"
    train_split_dir = f"{train_pair_clean}_splitted_w5_s2"
    model_name = f"chart_classification_model_{train_pair_clean}.h5"
    
    print(f"\n{'='*60}")
    print("TRAIN-TEST DIFFERENT PAIRS WORKFLOW")
    print(f"{'='*60}")
    print(f"Training on: {train_pair_name} ({train_csv})")
    print(f"Testing on:  {test_pair_name} ({test_csv})")
    print(f"Training images directory: {train_images_dir}")
    print(f"Training split directory: {train_split_dir}")
    print(f"Model name: {model_name}")
    print(f"{'='*60}\n")
    
    # Step 1: Generate training images
    print("\nStep 1: Generating training images...")
    if os.path.exists(train_images_dir):
        response = input(f"Directory {train_images_dir} already exists. Delete and regenerate? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(train_images_dir, ignore_errors=True)
        else:
            print(f"Using existing images in {train_images_dir}")
    
    if not os.path.exists(train_images_dir):
        cmd = [sys.executable, "Pattern_Technical indicators _Next Close price.py", train_csv_path, train_images_dir]
        if not run_command(cmd, "Generating training images"):
            sys.exit(1)
    
    # Step 2: Train the model
    print("\nStep 2: Training CNN model...")
    if os.path.exists(model_name):
        response = input(f"Model {model_name} already exists. Retrain? (y/n): ")
        if response.lower() != 'y':
            print(f"Using existing model: {model_name}")
        else:
            cmd = [sys.executable, "CNN.py", train_images_dir, train_split_dir, model_name]
            if not run_command(cmd, "Training CNN model"):
                sys.exit(1)
    else:
        cmd = [sys.executable, "CNN.py", train_images_dir, train_split_dir, model_name]
        if not run_command(cmd, "Training CNN model"):
            sys.exit(1)
    
    # Step 3: Test on different pair
    print("\nStep 3: Testing on different pair...")
    # Temporarily rename the model so simulate_forex_pair.py can use it
    temp_model_name = "chart_classification_model.h5"
    if os.path.exists(temp_model_name):
        backup_name = temp_model_name + ".backup"
        if os.path.exists(backup_name):
            os.remove(backup_name)
        os.rename(temp_model_name, backup_name)
    
    # Copy the trained model to the default name
    import shutil
    shutil.copy(model_name, temp_model_name)
    
    try:
        cmd = [sys.executable, "simulate_forex_pair.py", test_csv_path, test_pair_name]
        if test_first_date:
            cmd.append(test_first_date)
        if test_last_date:
            cmd.append(test_last_date)
        
        if not run_command(cmd, f"Testing on {test_pair_name}"):
            sys.exit(1)
    finally:
        # Restore original model if it existed
        if os.path.exists(backup_name):
            os.remove(temp_model_name)
            os.rename(backup_name, temp_model_name)
        elif os.path.exists(temp_model_name):
            os.remove(temp_model_name)
    
    # Step 4: Calculate metrics
    result_file = f'trade_result_{test_pair_clean}.txt'
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

