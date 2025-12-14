# -*- coding: utf-8 -*-
"""
Improved Chart Labeling Script for Profitable Trading Signals

Key improvements over original labeling:
1. Uses lookahead window (10 candles) instead of just next candle
2. Requires minimum profit threshold (0.15% / 15 pips) to label
3. Labels based on which direction has more profit potential
4. Skips unclear/noisy patterns that don't have clear directional bias

Usage:
    python generate_profitable_labels.py <csv_file> <output_dir> [first_date] [last_date]
    python generate_profitable_labels.py EURUSD_M15.csv chart_images_profitable
"""

import talib
import pandas as pd
import mplfinance as mpf
import os
import matplotlib.pyplot as plt
import sys

# ============================================================================
# CONFIGURATION - Adjust these for different trading styles
# ============================================================================
LOOKAHEAD_CANDLES = 10      # How many candles to look ahead (10 x 15min = 2.5 hours)
MIN_PROFIT_THRESHOLD = 0.0015  # Minimum 0.15% move required (15 pips on EUR/USD)
WINDOW_SIZE = 5              # Chart window size (candles to show)
SHIFT_SIZE = 2               # Step size between windows

# ============================================================================
# Load CSV data
# ============================================================================
if len(sys.argv) > 1:
    csv_file = sys.argv[1]
    if not os.path.isabs(csv_file) and '/' not in csv_file and '\\' not in csv_file:
        csv_file = os.path.join(os.getcwd(), 'data-cache', csv_file)
    elif not os.path.exists(csv_file):
        csv_file = os.path.join(os.getcwd(), 'data-cache', os.path.basename(csv_file))
else:
    csv_file = os.path.join(os.getcwd(), 'data-cache', 'EURUSD_M15.csv')

print(f"Using CSV file: {csv_file}")
print(f"Lookahead candles: {LOOKAHEAD_CANDLES}")
print(f"Min profit threshold: {MIN_PROFIT_THRESHOLD*100:.2f}%")

# Try different delimiters
try:
    data = pd.read_csv(csv_file, delimiter='\t', index_col='Time', parse_dates=True)
except:
    try:
        data = pd.read_csv(csv_file, delimiter=',', index_col='Time', parse_dates=True)
    except:
        data = pd.read_csv(csv_file, delimiter=',', index_col=0, parse_dates=True,
                          names=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
        data.index.name = 'Time'

if not isinstance(data.index, pd.DatetimeIndex):
    data.index = pd.to_datetime(data.index)
data = data.sort_index()

# Date range filtering
if len(sys.argv) > 3:
    first_date = sys.argv[3]
    last_date = sys.argv[4] if len(sys.argv) > 4 else None
    
    print(f"Filtering data from {first_date} to {last_date or 'end'}")
    first_dt = pd.to_datetime(first_date)
    if last_date:
        last_dt = pd.to_datetime(last_date)
        data = data[(data.index >= first_dt) & (data.index <= last_dt)]
    else:
        data = data[data.index >= first_dt]
    
    print(f"Filtered data range: {data.index[0]} to {data.index[-1]}")

print(f"Total data rows: {len(data)}")
data['SMA'] = talib.SMA(data['Close'], timeperiod=20)

# ============================================================================
# Pattern detection functions (same as original)
# ============================================================================
pattern_funcs = [
    ("Two Crows", talib.CDL2CROWS),
    ("Three Black Crows", talib.CDL3BLACKCROWS),
    ("Three Inside Up/Down", talib.CDL3INSIDE),
    ("Three-Line Strike", talib.CDL3LINESTRIKE),
    ("Three Outside Up/Down", talib.CDL3OUTSIDE),
    ("Three Stars In The South", talib.CDL3STARSINSOUTH),
    ("Three Advancing White Soldiers", talib.CDL3WHITESOLDIERS),
    ("Abandoned Baby", talib.CDLABANDONEDBABY),
    ("Advance Block", talib.CDLADVANCEBLOCK),
    ("Belt-hold", talib.CDLBELTHOLD),
    ("Breakaway", talib.CDLBREAKAWAY),
    ("Closing Marubozu", talib.CDLCLOSINGMARUBOZU),
    ("Concealing Baby Swallow", talib.CDLCONCEALBABYSWALL),
    ("Counterattack", talib.CDLCOUNTERATTACK),
    ("Dark Cloud Cover", talib.CDLDARKCLOUDCOVER),
    ("Doji", talib.CDLDOJI),
    ("Doji Star", talib.CDLDOJISTAR),
    ("Dragonfly Doji", talib.CDLDRAGONFLYDOJI),
    ("Engulfing Pattern", talib.CDLENGULFING),
    ("Evening Doji Star", talib.CDLEVENINGDOJISTAR),
    ("Evening Star", talib.CDLEVENINGSTAR),
    ("Up/Down-gap side-by-side white lines", talib.CDLGAPSIDESIDEWHITE),
    ("Gravestone Doji", talib.CDLGRAVESTONEDOJI),
    ("Hammer", talib.CDLHAMMER),
    ("Hanging Man", talib.CDLHANGINGMAN),
    ("Harami Pattern", talib.CDLHARAMI),
    ("Harami Cross Pattern", talib.CDLHARAMICROSS),
    ("High-Wave Candle", talib.CDLHIGHWAVE),
    ("Hikkake Pattern", talib.CDLHIKKAKE),
    ("Modified Hikkake Pattern", talib.CDLHIKKAKEMOD),
    ("Homing Pigeon", talib.CDLHOMINGPIGEON),
    ("Identical Three Crows", talib.CDLIDENTICAL3CROWS),
    ("In-Neck Pattern", talib.CDLINNECK),
    ("Inverted Hammer", talib.CDLINVERTEDHAMMER),
    ("Kicking", talib.CDLKICKING),
    ("Kicking - bull/bear determined by the longer marubozu", talib.CDLKICKINGBYLENGTH),
    ("Ladder Bottom", talib.CDLLADDERBOTTOM),
    ("Long Legged Doji", talib.CDLLONGLEGGEDDOJI),
    ("Long Line Candle", talib.CDLLONGLINE),
    ("Marubozu", talib.CDLMARUBOZU),
    ("Matching Low", talib.CDLMATCHINGLOW),
    ("Mat Hold", talib.CDLMATHOLD),
    ("Morning Doji Star", talib.CDLMORNINGDOJISTAR),
    ("Morning Star", talib.CDLMORNINGSTAR),
    ("On-Neck Pattern", talib.CDLONNECK),
    ("Piercing Pattern", talib.CDLPIERCING),
    ("Rickshaw Man", talib.CDLRICKSHAWMAN),
    ("Rising/Falling Three Methods", talib.CDLRISEFALL3METHODS),
    ("Separating Lines", talib.CDLSEPARATINGLINES),
    ("Shooting Star", talib.CDLSHOOTINGSTAR),
    ("Short Line Candle", talib.CDLSHORTLINE),
    ("Spinning Top", talib.CDLSPINNINGTOP),
    ("Stalled Pattern", talib.CDLSTALLEDPATTERN),
    ("Stick Sandwich", talib.CDLSTICKSANDWICH),
    ("Takuri", talib.CDLTAKURI),
    ("Tasuki Gap", talib.CDLTASUKIGAP),
    ("Thrusting Pattern", talib.CDLTHRUSTING),
    ("Tristar Pattern", talib.CDLTRISTAR),
    ("Unique 3 River", talib.CDLUNIQUE3RIVER),
    ("Upside Gap Two Crows", talib.CDLUPSIDEGAP2CROWS),
    ("Upside/Downside Gap Three Methods", talib.CDLXSIDEGAP3METHODS)
]

# ============================================================================
# Create output directories
# ============================================================================
if len(sys.argv) > 2:
    output_dir = sys.argv[2]
else:
    output_dir = "chart_images_profitable"

os.makedirs(output_dir, exist_ok=True)
uptrend_dir = os.path.join(output_dir, "uptrend")
downtrend_dir = os.path.join(output_dir, "downtrend")
os.makedirs(uptrend_dir, exist_ok=True)
os.makedirs(downtrend_dir, exist_ok=True)

# ============================================================================
# Generate labeled images with IMPROVED labeling logic
# ============================================================================
uptrend_count = 0
downtrend_count = 0
skipped_no_pattern = 0
skipped_insufficient_data = 0
skipped_below_threshold = 0
skipped_unclear = 0

total_iterations = (len(data) - WINDOW_SIZE - LOOKAHEAD_CANDLES) // SHIFT_SIZE
print(f"\nProcessing {total_iterations} windows...")

for i in range(0, len(data) - WINDOW_SIZE - LOOKAHEAD_CANDLES, SHIFT_SIZE):
    window = data.iloc[i:i+WINDOW_SIZE]
    last_candle_close = window['Close'].iloc[-1]

    # Check for candlestick pattern
    pattern_detected = False
    for name, func in pattern_funcs:
        if func(window['Open'], window['High'], window['Low'], window['Close']).iloc[-1] != 0:
            pattern_detected = True
            break

    if not pattern_detected:
        skipped_no_pattern += 1
        continue

    # Get future price data (LOOKAHEAD_CANDLES ahead)
    future_start = i + WINDOW_SIZE
    future_end = future_start + LOOKAHEAD_CANDLES

    if future_end > len(data):
        skipped_insufficient_data += 1
        continue

    future_closes = data['Close'].iloc[future_start:future_end]
    future_highs = data['High'].iloc[future_start:future_end]
    future_lows = data['Low'].iloc[future_start:future_end]

    # Calculate maximum profit potential in each direction
    # For LONG: max high in future - entry price
    # For SHORT: entry price - min low in future
    max_future_high = future_highs.max()
    min_future_low = future_lows.min()

    upside_potential = (max_future_high - last_candle_close) / last_candle_close
    downside_potential = (last_candle_close - min_future_low) / last_candle_close

    # Determine label based on profit potential
    timestamp_str = str(window.iloc[-1].name).replace(':', '-')

    # Both directions must exceed threshold, and one must be clearly better
    if upside_potential >= MIN_PROFIT_THRESHOLD and downside_potential >= MIN_PROFIT_THRESHOLD:
        # Both directions have potential - choose the stronger one
        # Require at least 1.5x advantage to avoid ambiguous signals
        if upside_potential > downside_potential * 1.5:
            label = 'uptrend'
            save_path = os.path.join(uptrend_dir, f"{timestamp_str}.png")
        elif downside_potential > upside_potential * 1.5:
            label = 'downtrend'
            save_path = os.path.join(downtrend_dir, f"{timestamp_str}.png")
        else:
            skipped_unclear += 1
            continue
    elif upside_potential >= MIN_PROFIT_THRESHOLD:
        label = 'uptrend'
        save_path = os.path.join(uptrend_dir, f"{timestamp_str}.png")
    elif downside_potential >= MIN_PROFIT_THRESHOLD:
        label = 'downtrend'
        save_path = os.path.join(downtrend_dir, f"{timestamp_str}.png")
    else:
        skipped_below_threshold += 1
        continue

    # Generate and save the chart image
    try:
        ap = [mpf.make_addplot(window['SMA'], color='blue', secondary_y=False)]
        mpf.plot(window, type='candle', style='yahoo', addplot=ap, volume=True,
                axisoff=True, ylabel='', savefig=save_path)
        plt.close()

        if label == 'uptrend':
            uptrend_count += 1
        else:
            downtrend_count += 1

    except Exception as e:
        print(f"Error saving chart: {e}")
        continue

    # Progress indicator
    if (uptrend_count + downtrend_count) % 500 == 0:
        print(f"Generated {uptrend_count + downtrend_count} images...")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*60)
print("LABELING COMPLETE - SUMMARY")
print("="*60)
print(f"Total uptrend images:   {uptrend_count}")
print(f"Total downtrend images: {downtrend_count}")
print(f"Total labeled images:   {uptrend_count + downtrend_count}")
print("-"*60)
print(f"Skipped (no pattern):        {skipped_no_pattern}")
print(f"Skipped (insufficient data): {skipped_insufficient_data}")
print(f"Skipped (below threshold):   {skipped_below_threshold}")
print(f"Skipped (unclear direction): {skipped_unclear}")
print("-"*60)
print(f"Output directory: {output_dir}")
print(f"Lookahead candles: {LOOKAHEAD_CANDLES}")
print(f"Min profit threshold: {MIN_PROFIT_THRESHOLD*100:.2f}%")
print("="*60)

# Balance check
if uptrend_count > 0 and downtrend_count > 0:
    ratio = max(uptrend_count, downtrend_count) / min(uptrend_count, downtrend_count)
    if ratio > 2:
        print(f"\n⚠️  WARNING: Class imbalance detected (ratio: {ratio:.1f}:1)")
        print("Consider adjusting threshold or using class weights during training.")
else:
    print("\n⚠️  WARNING: One or both classes have 0 samples!")

