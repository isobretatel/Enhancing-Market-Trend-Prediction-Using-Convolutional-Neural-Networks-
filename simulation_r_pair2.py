#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation for another trading pair/date range
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from mplfinance.original_flavor import candlestick_ohlc
import talib
import mplfinance as mpf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img
import tensorflow as tf
import shutil

def find_time_index(time,df):

        df=pd.DataFrame(df)
        index = df.index.get_loc(time)
        print(index)
        return index
    
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
print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")

# Use different date range for second pair (November 2024)
dd = os.path.join(os.getcwd(), 'EURUSD_M15-test.csv')
firstDate="2024-11-01 00:00:00"
lastDate="2024-11-18 06:45:00"

data = pd.read_csv(dd, delimiter=',', index_col='Time', parse_dates=True)
data['SMA'] = talib.SMA(data['Close'], timeperiod=20)
initialTime_index=find_time_index(firstDate,data)
finalTime_index=find_time_index(lastDate,data)
data=data[initialTime_index:finalTime_index]

output_dir = "test_for_signal_pair2"
shutil.rmtree(output_dir,ignore_errors=True)
os.makedirs(output_dir, exist_ok=True)
window_size=5
shift_size=2
for i in range(0, len(data) - window_size,shift_size):
    window = data.iloc[i:i+window_size]
    timestamp_str = str(window.iloc[-1].name).replace(':', '-')
    save_path = os.path.join(output_dir, f"{timestamp_str}.png")
    ap = [mpf.make_addplot(window['SMA'], color='blue', secondary_y=False)]
    mpf.plot(window, type='candle', style='yahoo', addplot=ap, volume=True, axisoff=True, ylabel='',
 savefig=save_path)
    plt.close()

# Create DataFrame
data = pd.read_csv(dd, delimiter=',', parse_dates=True)
data=data[initialTime_index:finalTime_index]
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Time'])
df['Date'] = df['Date'].map(mdates.date2num)

# Create subplots and plot candlestick chart
fig, ax = plt.subplots(figsize=(14, 7))
candlestick_ohlc(ax, df[['Date', 'Open', 'High', 'Low', 'Close']].values, width=0.01, colorup='green', colordown='red')

# This part makes predictions 
from tensorflow.keras.utils import img_to_array

dataset_path="test_for_signal_pair2"
X=[]
for name in os.listdir(dataset_path):
    image1 = load_img(os.path.join(dataset_path, name), color_mode = 'rgb', interpolation="bilinear",target_size = (150, 150))
    image1 = img_to_array(image1)
    image1 = image1 / 255
    X.append(image1)
X=np.array(X) 
model=load_model("chart_classification_model.h5")
predictions = model.predict(X)

image_names=os.listdir(dataset_path)
indicator_xcoordinates=[]
indicator_trends=[]
for idx,i in enumerate(predictions):
    if i>=0.5:
        indicator_xcoordinates.append(os.path.splitext(image_names[idx])[0])
        indicator_trends.append("U")
    else:
        indicator_xcoordinates.append(os.path.splitext(image_names[idx])[0])
        indicator_trends.append("D")

## remove consecutive the same signals
signal_x = [indicator_xcoordinates[0]]
signal_label = [indicator_trends[0]] 
for i in range(1, len(indicator_trends)):
    if indicator_trends[i] != indicator_trends[i - 1]:
        signal_x.append(indicator_xcoordinates[i])
        signal_label.append(indicator_trends[i])
        
indicator_xcoordinates=signal_x
indicator_trends=signal_label  

# Add annotations for up/down labels
for time, label in zip(indicator_xcoordinates, indicator_trends):
    time_str = str(time)
    if ' ' in time_str and '-' in time_str.split(' ')[1]:
        date_part, time_part = time_str.split(' ', 1)
        time_part = time_part.replace('-', ':')
        time = f"{date_part} {time_part}"
    
    if time in df['Time'].values:
        result = df.isin([time])
        locations = result.stack()[result.stack()] 
        row = locations.index[0][0]
        row=df.loc[row]
        timestamp = mdates.date2num(pd.to_datetime(time))
        if label=='D':
            y_position = row['High'] + 0.00022
        else:
            y_position = row['Low'] - 0.00032
        ax.annotate(label,
                    xy=(timestamp, y_position),
                    xytext=(0, 2),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', 
                             fc='yellow', 
                             alpha=0.5)
                    )
    else:
            print(f"Time {time} not found in data")

ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d,%H:%M'))
plt.title('Up/Down Trend Signals for EUR/USD (Pair 2: Jan-Feb 2024)')
plt.xticks(rotation=60)
plt.xlabel('Date-Time')
plt.ylabel('Price')
plt.grid(True)
plt.tight_layout()
plt.show()

initial_amount_usd = 1000
current_amount_usd = initial_amount_usd
amount_in_euros = 0
number_changes=0

# Open file to write trading results
with open('trade_result_pair2.txt', 'w') as f:
    f.write(f"\nInitial amount in Dollar: {initial_amount_usd:.2f}\n")
    
    for i in range(len(indicator_xcoordinates)):
        time=indicator_xcoordinates[i]
        
        time_str = str(time)
        if ' ' in time_str and '-' in time_str.split(' ')[1]:
            date_part, time_part = time_str.split(' ', 1)
            time_part = time_part.replace('-', ':')
            time = f"{date_part} {time_part}"
        
        if time in df['Time'].values:
            result = df.isin([time])
            locations = result.stack()[result.stack()] 
            row = locations.index[0][0]
            row=df.loc[row]
            
            if indicator_trends[i] == 'U' and current_amount_usd > 0:
                amount_in_euros = current_amount_usd / row['Open']
                number_changes+=1
                current_amount_usd = 0
                f.write(f"Bought at {time} at price {row['Open']}, amount in euros: {amount_in_euros:.2f}\n")
                print(f"Bought at {time} at price {row['Open']}, amount in euros: {amount_in_euros:.2f}")
                
            elif indicator_trends[i] == 'D' and amount_in_euros > 0:
                current_amount_usd = amount_in_euros * row['Open']
                amount_in_euros = 0
                f.write(f"Sold at {time} at price {row['Open']}, amount in USD: {current_amount_usd:.2f}\n")
                print(f"Sold at {time} at price {row['Open']}, amount in USD: {current_amount_usd:.2f}")
                number_changes+=1

    if amount_in_euros > 0:
        f.write(f"\nFinal amount in EUR: {amount_in_euros:.2f}\n")
        print(f"\nFinal amount in EUR: {amount_in_euros:.2f}")
        current_amount_usd=amount_in_euros*data['Open'][finalTime_index-1]
        f.write(f"Final amount in Dollar: {current_amount_usd:.2f}\n")
        print(f"\nFinal amount in Dollar: {current_amount_usd:.2f}")
    else:
        f.write(f"Final amount in Dollar: {current_amount_usd:.2f}\n")
        print(f"\nFinal amount in Dollar: {current_amount_usd:.2f}")
        
    f.write(f"Total number of buy/sell: {number_changes}\n")
    print(f"\nTotal number of buy/sell: {number_changes}")

