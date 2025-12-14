# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 12:27:20 2024

@author: edree
"""

import talib
import pandas as pd
import mplfinance as mpf
import os
import matplotlib.pyplot as plt
import sys

# Allow CSV file to be specified via command line argument
if len(sys.argv) > 1:
    csv_file = sys.argv[1]
    # If just filename provided, look in data-cache
    if not os.path.isabs(csv_file) and '/' not in csv_file and '\\' not in csv_file:
        csv_file = os.path.join(os.getcwd(), 'data-cache', csv_file)
    elif not os.path.exists(csv_file):
        csv_file = os.path.join(os.getcwd(), 'data-cache', os.path.basename(csv_file))
else:
    # Default to EURUSD_M15.csv
    csv_file = os.path.join(os.getcwd(), 'data-cache', 'EURUSD_M15.csv')

print(f"Using CSV file: {csv_file}")

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

# Ensure index is datetime
if not isinstance(data.index, pd.DatetimeIndex):
    data.index = pd.to_datetime(data.index)
data = data.sort_index()

# Allow date range filtering via command line arguments
if len(sys.argv) > 3:
    first_date = sys.argv[3]
    if len(sys.argv) > 4:
        last_date = sys.argv[4]
    else:
        last_date = None
    
    print(f"Filtering data from {first_date} to {last_date or 'end'}")
    first_dt = pd.to_datetime(first_date)
    if last_date:
        last_dt = pd.to_datetime(last_date)
        data = data[(data.index >= first_dt) & (data.index <= last_dt)]
    else:
        data = data[data.index >= first_dt]
    
    print(f"Filtered data range: {data.index[0]} to {data.index[-1]}")
    print(f"Filtered data rows: {len(data)}")

data['SMA'] = talib.SMA(data['Close'], timeperiod=20)

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
    ("Takuri (Dragonfly Doji with very long lower shadow)", talib.CDLTAKURI),
    ("Tasuki Gap", talib.CDLTASUKIGAP),
    ("Thrusting Pattern", talib.CDLTHRUSTING),
    ("Tristar Pattern", talib.CDLTRISTAR),
    ("Unique 3 River", talib.CDLUNIQUE3RIVER),
    ("Upside Gap Two Crows", talib.CDLUPSIDEGAP2CROWS),
    ("Upside/Downside Gap Three Methods", talib.CDLXSIDEGAP3METHODS)
]

# Allow output directory to be specified via command line argument
if len(sys.argv) > 2:
    output_dir = sys.argv[2]
else:
    # Default output directory
    output_dir = "chart_images5_1"
os.makedirs(output_dir, exist_ok=True)
uptrend_dir = os.path.join(output_dir, "uptrend")
downtrend_dir = os.path.join(output_dir, "downtrend")
os.makedirs(uptrend_dir, exist_ok=True)
os.makedirs(downtrend_dir, exist_ok=True)
window_size=5
shift_size=2
for i in range(0, len(data) - window_size,shift_size):
    window = data.iloc[i:i+window_size]
    next_candle_close = data['Close'].iloc[i+window_size]
    last_candle_close = window['Close'].iloc[-1]
    sma_last = window['SMA'].iloc[-1]

    pattern_detected = False

    for name, func in pattern_funcs:
        if func(window['Open'], window['High'], window['Low'], window['Close']).iloc[-1] != 0:
            pattern_detected = True
            break


    if pattern_detected:
        # Label based on FUTURE price movement
        if next_candle_close > last_candle_close:
            label = 'uptrend'
            # Use timestamp instead of index for chronological ordering
            timestamp_str = str(window.iloc[-1].name).replace(':', '-')
            save_path = os.path.join(uptrend_dir, f"{timestamp_str}.png")
        elif next_candle_close < last_candle_close:
            label = 'downtrend'
            # Use timestamp instead of index for chronological ordering
            timestamp_str = str(window.iloc[-1].name).replace(':', '-')
            save_path = os.path.join(downtrend_dir, f"{timestamp_str}.png")
        else:
            continue

        ap = [mpf.make_addplot(window['SMA'], color='blue', secondary_y=False)]
        mpf.plot(window, type='candle', style='yahoo', addplot=ap, volume=True, axisoff=True, ylabel='',
 savefig=save_path)
        plt.close()

