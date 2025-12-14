#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run train/test workflow for all available forex pairs
"""

import subprocess
import sys
import os

# Define pairs: (csv_file, pair_name)
pairs = [
    ("EURUSD_M15.csv", "EUR/USD"),
    ("GBPUSD15.csv", "GBP/USD"),
    ("USDCAD15.csv", "USD/CAD"),
    ("USDJPY15.csv", "USD/JPY"),
]

def run_train_test(csv_file, pair_name):
    """Run train/test workflow for a single pair"""
    print(f"\n{'='*80}")
    print(f"Processing: {pair_name} ({csv_file})")
    print(f"{'='*80}\n")
    
    cmd = [
        sys.executable,
        "train_test_same_pair.py",
        csv_file,
        pair_name,
        "--2y-1y"
    ]
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0

def main():
    print("="*80)
    print("RUNNING TRAIN/TEST FOR ALL FOREX PAIRS")
    print("="*80)
    print(f"Total pairs to process: {len(pairs)}")
    print("="*80)
    
    results = {}
    
    for csv_file, pair_name in pairs:
        csv_path = os.path.join('data-cache', csv_file)
        if not os.path.exists(csv_path):
            print(f"\nWARNING: {csv_file} not found, skipping {pair_name}")
            results[pair_name] = False
            continue
        
        success = run_train_test(csv_file, pair_name)
        results[pair_name] = success
        
        if not success:
            print(f"\nERROR: Failed to process {pair_name}")
        else:
            print(f"\nSUCCESS: Completed {pair_name}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for pair_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{pair_name:20} {status}")
    print("="*80)

if __name__ == "__main__":
    main()

