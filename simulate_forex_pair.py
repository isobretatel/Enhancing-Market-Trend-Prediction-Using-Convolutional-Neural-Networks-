#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic Forex Pair Simulation Script
Can work with any forex pair CSV file
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import talib
import mplfinance as mpf
import os
import numpy as np
import shutil
import sys
import re

# Fix Windows console encoding for Unicode characters (e.g., checkmarks from TensorFlow)
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import tensorflow as tf

def find_time_index(time, df):
    """Find the index of a given timestamp"""
    try:
        index = df.index.get_loc(time)
        return index
    except KeyError:
        # Find nearest timestamp
        try:
            index = df.index.searchsorted(pd.to_datetime(time))
            return min(index, len(df) - 1)
        except:
            return 0

# Confidence threshold - only trade when model prediction exceeds this threshold
# Higher threshold = fewer trades but higher confidence
CONFIDENCE_THRESHOLD = 0.7  # Default: 0.7 (70% confidence required to trade)

def simulate_forex_pair(csv_file, pair_name, first_date=None, last_date=None, output_dir_suffix="", confidence_threshold=None):
    # Ensure csv_file path is correct - check data-cache if not absolute path
    if not os.path.isabs(csv_file):
        if not os.path.exists(csv_file):
            data_cache_path = os.path.join('data-cache', csv_file)
            if os.path.exists(data_cache_path):
                csv_file = data_cache_path
        elif not csv_file.startswith('data-cache') and os.path.exists(os.path.join('data-cache', os.path.basename(csv_file))):
            csv_file = os.path.join('data-cache', os.path.basename(csv_file))
    """Simulate trading for a forex pair"""
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus: 
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=7492)]
            )
        except RuntimeError as e:
            print(f"GPU configuration warning: {e}")

    logical_gpus = tf.config.list_logical_devices('GPU')
    print(f"{len(gpus)} Physical GPU, {len(logical_gpus)} Logical GPUs")
    
    # Read data
    print(f"\nLoading {pair_name} data from {csv_file}...")
    # Try tab delimiter first (common for MT4 exports)
    try:
        data = pd.read_csv(csv_file, delimiter='\t', index_col=0, parse_dates=True, 
                          names=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'], 
                          header=0, skiprows=0)
        # Check if first row is actually header
        if isinstance(data.index[0], str) and data.index[0] in ['Time', 'time', 'TIME']:
            data = pd.read_csv(csv_file, delimiter='\t', index_col=0, parse_dates=True, 
                              header=0)
    except:
        try:
            data = pd.read_csv(csv_file, delimiter=',', index_col='Time', parse_dates=True)
        except:
            try:
                data = pd.read_csv(csv_file, delimiter=',', index_col=0, parse_dates=True,
                                 header=0)
                # Check if column names are correct
                if 'Close' not in data.columns:
                    data = pd.read_csv(csv_file, delimiter=',', index_col=0, parse_dates=True,
                                     names=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'],
                                     header=0)
            except:
                # Last resort: read without header and assign names
                data = pd.read_csv(csv_file, delimiter=',', index_col=0, parse_dates=True,
                                 names=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'],
                                 header=None)
    
    print(f"Data loaded: {len(data)} rows")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Calculate SMA
    data['SMA'] = talib.SMA(data['Close'], timeperiod=20)
    
    # Determine date range
    if first_date is None:
        first_date = str(data.index[0])
    if last_date is None:
        last_date = str(data.index[-1])
    
    initialTime_index = find_time_index(first_date, data)
    finalTime_index = find_time_index(last_date, data)
    
    print(f"Using date range: {first_date} to {last_date}")
    print(f"Indices: {initialTime_index} to {finalTime_index}")
    
    data = data[initialTime_index:finalTime_index]
    
    # Create output directory
    output_dir = f"test_for_signal_{pair_name.lower().replace('/', '_')}{output_dir_suffix}"
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate chart images (only if directory doesn't exist or is empty)
    print(f"\nGenerating chart images...")
    window_size = 5
    shift_size = 2
    
    # Always regenerate images for the specified date range to ensure matching
    # Remove old images if they exist
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            if f.endswith('.png'):
                os.remove(os.path.join(output_dir, f))
    
    existing_images = 0
    
    if True:  # Always generate fresh images
        # Limit to reasonable number of images for performance (max 3000)
        max_images = 3000
        total_windows = len(data) - window_size
        if total_windows > max_images:
            step = max(1, total_windows // max_images)
            print(f"Limiting to {max_images} images (using step={step})")
        else:
            step = shift_size
        
        image_count = 0
        for i in range(0, len(data) - window_size, step):
            if image_count >= max_images:
                break
            window = data.iloc[i:i+window_size]
            timestamp_str = str(window.iloc[-1].name).replace(':', '-')
            save_path = os.path.join(output_dir, f"{timestamp_str}.png")
            ap = [mpf.make_addplot(window['SMA'], color='blue', secondary_y=False)]
            mpf.plot(window, type='candle', style='yahoo', addplot=ap, volume=True, 
                    axisoff=True, ylabel='', savefig=save_path)
            plt.close()
            image_count += 1
            if image_count % 500 == 0:
                print(f"Generated {image_count} images...")
        
        print(f"Generated {image_count} chart images")
    else:
        print(f"Using {existing_images} existing images")
    
    # Load model and make predictions
    print("\nLoading model and making predictions...")
    model = load_model("chart_classification_model.h5")
    
    X = []
    image_names = []
    for name in sorted(os.listdir(output_dir)):
        if name.endswith('.png'):
            image1 = load_img(os.path.join(output_dir, name), color_mode='rgb', 
                            interpolation="bilinear", target_size=(150, 150))
            image1 = img_to_array(image1)
            image1 = image1 / 255
            X.append(image1)
            image_names.append(name)
    
    X = np.array(X)
    predictions = model.predict(X, verbose=0)

    # Use confidence threshold (from parameter or global default)
    threshold = confidence_threshold if confidence_threshold is not None else CONFIDENCE_THRESHOLD
    print(f"Using confidence threshold: {threshold}")

    # Process predictions - only include signals that exceed confidence threshold
    indicator_xcoordinates = []
    indicator_trends = []
    skipped_low_confidence = 0
    for idx, pred in enumerate(predictions):
        timestamp = os.path.splitext(image_names[idx])[0]
        # pred >= threshold means confident UP, pred <= (1-threshold) means confident DOWN
        if pred >= threshold:
            indicator_trends.append("U")
            indicator_xcoordinates.append(timestamp)
        elif pred <= (1 - threshold):
            indicator_trends.append("D")
            indicator_xcoordinates.append(timestamp)
        else:
            skipped_low_confidence += 1

    print(f"Skipped {skipped_low_confidence} low-confidence predictions")

    # Handle case where no signals pass the threshold
    if len(indicator_xcoordinates) == 0:
        print("WARNING: No signals passed the confidence threshold!")
        print("Try lowering the threshold or check model quality.")
        return None

    # Remove consecutive same signals
    signal_x = [indicator_xcoordinates[0]]
    signal_label = [indicator_trends[0]]
    for i in range(1, len(indicator_trends)):
        if indicator_trends[i] != indicator_trends[i - 1]:
            signal_x.append(indicator_xcoordinates[i])
            signal_label.append(indicator_trends[i])

    indicator_xcoordinates = signal_x
    indicator_trends = signal_label

    print(f"Generated {len(indicator_xcoordinates)} trading signals (after filtering)")
    
    # Prepare data for trading simulation - use the same data we already loaded
    # Create dataframe from the filtered data
    df = data.reset_index()
    df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'SMA']
    df['Date'] = pd.to_datetime(df['Time'])
    df['Date'] = df['Date'].map(mdates.date2num)
    
    # Trading simulation
    initial_amount_usd = 1000
    current_amount_usd = initial_amount_usd
    amount_in_base = 0  # Amount in base currency (GBP, JPY, CAD, etc.)
    number_changes = 0
    
    # Determine base currency from pair name
    base_currency = pair_name.split('/')[0] if '/' in pair_name else 'EUR'
    
    output_file = f'trade_result_{pair_name.lower().replace("/", "_")}{output_dir_suffix}.txt'
    
    with open(output_file, 'w') as f:
        f.write(f"\nInitial amount in Dollar: {initial_amount_usd:.2f}\n")
        f.write(f"Trading Pair: {pair_name}\n\n")
        
        # Prepare time matching - create lookup dictionary for fast access
        df['Time_dt'] = pd.to_datetime(df['Time'])
        df['Time_str'] = df['Time_dt'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df['Time_str_filename'] = df['Time_dt'].dt.strftime('%Y-%m-%d %H-%M-%S')  # Windows filename format
        
        # Create lookup dictionaries
        time_lookup = {}
        for idx, row in df.iterrows():
            time_lookup[row['Time_str']] = row
            time_lookup[row['Time_str_filename']] = row
        
        print(f"\nMatching {len(indicator_xcoordinates)} signals with {len(df)} data points...")
        matched_count = 0
        
        for i in range(len(indicator_xcoordinates)):
            time_filename = indicator_xcoordinates[i]
            
            # Convert filename timestamp to datetime format
            # Filename format: "2024-01-01 12-30-00" (dashes) -> "2024-01-01 12:30:00" (colons)
            time_str = str(time_filename)
            if ' ' in time_str and '-' in time_str.split(' ')[1]:
                date_part, time_part = time_str.split(' ', 1)
                time_part = time_part.replace('-', ':')
                time_normalized = f"{date_part} {time_part}"
            else:
                time_normalized = time_str
            
            # Try multiple matching strategies
            row = None
            
            # Strategy 1: Direct lookup from dictionaries
            if time_filename in time_lookup:
                row = time_lookup[time_filename]
                matched_count += 1
            elif time_normalized in time_lookup:
                row = time_lookup[time_normalized]
                matched_count += 1
            else:
                # Strategy 2: Find closest timestamp
                try:
                    time_dt = pd.to_datetime(time_normalized)
                    time_diffs = (df['Time_dt'] - time_dt).abs()
                    closest_idx = time_diffs.idxmin()
                    time_diff_seconds = time_diffs.loc[closest_idx].total_seconds()
                    
                    # Accept if within 15 minutes (900 seconds)
                    if time_diff_seconds < 900:
                        row = df.loc[closest_idx]
                        matched_count += 1
                except Exception as e:
                    continue  # Skip this signal if can't parse
            
            if row is not None:
                # CRITICAL: Avoid data leak - execute trade at NEXT candle's open price
                # Signal is generated at time T, but we can only execute at time T+1's open
                row_idx = df.index.get_loc(row.name) if hasattr(row, 'name') and row.name in df.index else None
                if row_idx is not None and row_idx + 1 < len(df):
                    execution_row = df.iloc[row_idx + 1]
                    execution_price = execution_row['Open']
                    execution_time = execution_row['Time']
                else:
                    continue  # Skip if can't get next candle
                
                if indicator_trends[i] == 'U' and current_amount_usd > 0:  # Buy signal
                    amount_in_base = current_amount_usd / execution_price
                    number_changes += 1
                    current_amount_usd = 0
                    f.write(f"Bought at {execution_time} at price {execution_price}, amount in {base_currency}: {amount_in_base:.2f}\n")
                    if number_changes <= 10:
                        print(f"Bought at {execution_time} at price {execution_price}, amount in {base_currency}: {amount_in_base:.2f}")
                    
                elif indicator_trends[i] == 'D' and amount_in_base > 0:  # Sell signal
                    current_amount_usd = amount_in_base * execution_price
                    amount_in_base = 0
                    f.write(f"Sold at {execution_time} at price {execution_price}, amount in USD: {current_amount_usd:.2f}\n")
                    if number_changes <= 10:
                        print(f"Sold at {execution_time} at price {execution_price}, amount in USD: {current_amount_usd:.2f}")
                    number_changes += 1
        
        print(f"Matched {matched_count} out of {len(indicator_xcoordinates)} signals")
        
        # Final amount
        if amount_in_base > 0:
            f.write(f"\nFinal amount in {base_currency}: {amount_in_base:.2f}\n")
            print(f"\nFinal amount in {base_currency}: {amount_in_base:.2f}")
            current_amount_usd = amount_in_base * data['Open'].iloc[-1]
            f.write(f"Final amount in Dollar: {current_amount_usd:.2f}\n")
            print(f"\nFinal amount in Dollar: {current_amount_usd:.2f}")
        else:
            f.write(f"Final amount in Dollar: {current_amount_usd:.2f}\n")
            print(f"\nFinal amount in Dollar: {current_amount_usd:.2f}")
        
        f.write(f"Total number of buy/sell: {number_changes}\n")
        print(f"\nTotal number of buy/sell: {number_changes}")
    
    print(f"\nResults saved to {output_file}")
    return output_file

if __name__ == "__main__":
    # Usage: python simulate_forex_pair.py <csv_file> <pair_name> [first_date] [last_date] [confidence_threshold]
    # Example: python simulate_forex_pair.py GBPUSD15.csv GBP/USD 2024-01-01 2024-06-01 0.75

    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        # If just filename provided, look in data-cache
        if not os.path.isabs(csv_file) and '/' not in csv_file and '\\' not in csv_file:
            if os.path.exists(os.path.join('data-cache', csv_file)):
                csv_file = os.path.join('data-cache', csv_file)
        pair_name = sys.argv[2] if len(sys.argv) > 2 else csv_file.replace('.csv', '').replace('_', '/').replace('data-cache/', '').replace('data-cache\\', '').upper()
    else:
        csv_file = os.path.join('data-cache', 'GBPUSD15.csv')
        pair_name = 'GBP/USD'

    # Get date range from command line or use defaults
    first_date = sys.argv[3] if len(sys.argv) > 3 else None
    last_date = sys.argv[4] if len(sys.argv) > 4 else None

    # Get confidence threshold from command line or use default
    confidence_threshold = float(sys.argv[5]) if len(sys.argv) > 5 else None

    result_file = simulate_forex_pair(csv_file, pair_name, first_date, last_date, confidence_threshold=confidence_threshold)
    if result_file:
        print(f"\nSimulation complete! Results saved to {result_file}")
    else:
        print("\nSimulation failed - no valid signals generated.")

