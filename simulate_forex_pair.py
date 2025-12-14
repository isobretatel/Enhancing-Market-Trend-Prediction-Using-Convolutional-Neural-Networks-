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
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import tensorflow as tf
import shutil
import sys
import re

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

def simulate_forex_pair(csv_file, pair_name, first_date=None, last_date=None, output_dir_suffix=""):
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
                          names=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    except:
        try:
            data = pd.read_csv(csv_file, delimiter=',', index_col='Time', parse_dates=True)
        except:
            data = pd.read_csv(csv_file, delimiter=',', index_col=0, parse_dates=True,
                             names=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    
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
    
    # Generate chart images
    print(f"\nGenerating chart images...")
    window_size = 5
    shift_size = 2
    
    for i in range(0, len(data) - window_size, shift_size):
        window = data.iloc[i:i+window_size]
        timestamp_str = str(window.iloc[-1].name).replace(':', '-')
        save_path = os.path.join(output_dir, f"{timestamp_str}.png")
        ap = [mpf.make_addplot(window['SMA'], color='blue', secondary_y=False)]
        mpf.plot(window, type='candle', style='yahoo', addplot=ap, volume=True, 
                axisoff=True, ylabel='', savefig=save_path)
        plt.close()
    
    print(f"Generated {len(os.listdir(output_dir))} chart images")
    
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
    
    # Process predictions
    indicator_xcoordinates = []
    indicator_trends = []
    for idx, pred in enumerate(predictions):
        timestamp = os.path.splitext(image_names[idx])[0]
        if pred >= 0.5:
            indicator_trends.append("U")
        else:
            indicator_trends.append("D")
        indicator_xcoordinates.append(timestamp)
    
    # Remove consecutive same signals
    signal_x = [indicator_xcoordinates[0]]
    signal_label = [indicator_trends[0]]
    for i in range(1, len(indicator_trends)):
        if indicator_trends[i] != indicator_trends[i - 1]:
            signal_x.append(indicator_xcoordinates[i])
            signal_label.append(indicator_trends[i])
    
    indicator_xcoordinates = signal_x
    indicator_trends = signal_label
    
    print(f"Generated {len(indicator_xcoordinates)} trading signals")
    
    # Prepare data for trading simulation
    try:
        data_for_trading = pd.read_csv(csv_file, delimiter='\t', index_col=0, parse_dates=True,
                                      names=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    except:
        try:
            data_for_trading = pd.read_csv(csv_file, delimiter=',', parse_dates=True)
        except:
            data_for_trading = pd.read_csv(csv_file, delimiter=',', index_col=0, parse_dates=True,
                                          names=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    
    # Reset index if Time is in index
    if data_for_trading.index.name == 'Time' or isinstance(data_for_trading.index, pd.DatetimeIndex):
        data_for_trading = data_for_trading.reset_index()
        if 'Time' not in data_for_trading.columns and len(data_for_trading.columns) > 0:
            data_for_trading.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    data_for_trading = data_for_trading[initialTime_index:finalTime_index]
    df = pd.DataFrame(data_for_trading)
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
        
        for i in range(len(indicator_xcoordinates)):
            time = indicator_xcoordinates[i]
            
            # Convert filename back to timestamp format
            time_str = str(time)
            if ' ' in time_str and '-' in time_str.split(' ')[1]:
                date_part, time_part = time_str.split(' ', 1)
                time_part = time_part.replace('-', ':')
                time = f"{date_part} {time_part}"
            
            if time in df['Time'].values:
                result = df.isin([time])
                locations = result.stack()[result.stack()]
                row_idx = locations.index[0][0]
                row = df.loc[row_idx]
                
                if indicator_trends[i] == 'U' and current_amount_usd > 0:  # Buy signal
                    amount_in_base = current_amount_usd / row['Open']
                    number_changes += 1
                    current_amount_usd = 0
                    f.write(f"Bought at {time} at price {row['Open']}, amount in {base_currency}: {amount_in_base:.2f}\n")
                    print(f"Bought at {time} at price {row['Open']}, amount in {base_currency}: {amount_in_base:.2f}")
                    
                elif indicator_trends[i] == 'D' and amount_in_base > 0:  # Sell signal
                    current_amount_usd = amount_in_base * row['Open']
                    amount_in_base = 0
                    f.write(f"Sold at {time} at price {row['Open']}, amount in USD: {current_amount_usd:.2f}\n")
                    print(f"Sold at {time} at price {row['Open']}, amount in USD: {current_amount_usd:.2f}")
                    number_changes += 1
        
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
    # Default: GBP/USD 15-minute
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        pair_name = sys.argv[2] if len(sys.argv) > 2 else csv_file.replace('.csv', '').replace('_', '/').upper()
    else:
        csv_file = 'GBPUSD15.csv'
        pair_name = 'GBP/USD'
    
    # Get date range from command line or use defaults
    first_date = sys.argv[3] if len(sys.argv) > 3 else None
    last_date = sys.argv[4] if len(sys.argv) > 4 else None
    
    result_file = simulate_forex_pair(csv_file, pair_name, first_date, last_date)
    print(f"\nSimulation complete! Results saved to {result_file}")

