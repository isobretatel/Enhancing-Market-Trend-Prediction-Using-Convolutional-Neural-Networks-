# -*- coding: utf-8 -*-
"""
Calculate trading performance metrics: Sharpe Ratio, Sortino Ratio, ROI, and Maximum Drawdown
"""

import pandas as pd
import numpy as np
import re

def parse_trade_results(file_path):
    """Parse trade_result.txt and extract trading transactions"""
    transactions = []
    initial_amount = None
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Extract initial amount
        if 'Initial amount in Dollar:' in line:
            initial_amount = float(re.search(r'(\d+\.\d+)', line).group(1))
            continue
        
        # Extract buy transactions
        if 'Bought at' in line:
            match = re.search(r'Bought at (.+?) at price ([\d.]+), amount in euros: ([\d.]+)', line)
            if match:
                timestamp = match.group(1)
                price = float(match.group(2))
                amount_eur = float(match.group(3))
                transactions.append({
                    'timestamp': timestamp,
                    'type': 'buy',
                    'price': price,
                    'amount_eur': amount_eur,
                    'amount_usd': None
                })
        
        # Extract sell transactions
        elif 'Sold at' in line:
            match = re.search(r'Sold at (.+?) at price ([\d.]+), amount in USD: ([\d.]+)', line)
            if match:
                timestamp = match.group(1)
                price = float(match.group(2))
                amount_usd = float(match.group(3))
                transactions.append({
                    'timestamp': timestamp,
                    'type': 'sell',
                    'price': price,
                    'amount_eur': None,
                    'amount_usd': amount_usd
                })
    
    return initial_amount, transactions

def calculate_returns(initial_amount, transactions):
    """Calculate returns for each trade"""
    returns = []
    current_usd = initial_amount
    
    i = 0
    while i < len(transactions):
        if transactions[i]['type'] == 'buy':
            # Find corresponding sell
            if i + 1 < len(transactions) and transactions[i + 1]['type'] == 'sell':
                buy_price = transactions[i]['price']
                sell_price = transactions[i + 1]['price']
                
                # Calculate return
                if current_usd > 0:
                    amount_eur = current_usd / buy_price
                    final_usd = amount_eur * sell_price
                    trade_return = (final_usd - current_usd) / current_usd
                    returns.append(trade_return)
                    current_usd = final_usd
                i += 2
            else:
                i += 1
        else:
            i += 1
    
    return returns

def calculate_equity_curve(initial_amount, transactions):
    """Calculate equity curve over time"""
    equity = [initial_amount]
    current_usd = initial_amount
    amount_eur = 0
    
    for trans in transactions:
        if trans['type'] == 'buy' and current_usd > 0:
            amount_eur = current_usd / trans['price']
            current_usd = 0
            equity.append(current_usd + amount_eur * trans['price'])
        elif trans['type'] == 'sell' and amount_eur > 0:
            current_usd = amount_eur * trans['price']
            amount_eur = 0
            equity.append(current_usd)
    
    return equity

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculate Sharpe Ratio"""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    # Annualize if we have enough data
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    # Assuming daily returns, annualize
    # For 15-minute data, we have ~96 periods per day, ~252 trading days per year
    periods_per_year = 96 * 252
    
    if len(returns) > 1:
        sharpe = (mean_return - risk_free_rate / periods_per_year) / std_return * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0
    
    return sharpe

def calculate_sortino_ratio(returns, risk_free_rate=0.0):
    """Calculate Sortino Ratio (only penalizes downside deviation)"""
    if len(returns) == 0:
        return 0.0
    
    # Calculate downside deviation (only negative returns)
    downside_returns = [r for r in returns if r < 0]
    
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return float('inf') if np.mean(returns) > risk_free_rate else 0.0
    
    mean_return = np.mean(returns)
    downside_std = np.std(downside_returns)
    
    # Annualize
    periods_per_year = 96 * 252
    
    if len(returns) > 1:
        sortino = (mean_return - risk_free_rate / periods_per_year) / downside_std * np.sqrt(periods_per_year)
    else:
        sortino = 0.0
    
    return sortino

def calculate_max_drawdown(equity_curve):
    """Calculate Maximum Drawdown"""
    if len(equity_curve) == 0:
        return 0.0
    
    peak = equity_curve[0]
    max_dd = 0.0
    
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > max_dd:
            max_dd = dd
    
    return max_dd

def calculate_roi(initial_amount, final_amount):
    """Calculate Return on Investment"""
    if initial_amount == 0:
        return 0.0
    return (final_amount - initial_amount) / initial_amount * 100

# Main execution
if __name__ == "__main__":
    import sys
    file_path = sys.argv[1] if len(sys.argv) > 1 else 'trade_result.txt'
    
    print("=" * 60)
    print("TRADING PERFORMANCE METRICS")
    print("=" * 60)
    
    # Parse trade results
    initial_amount, transactions = parse_trade_results(file_path)
    
    if initial_amount is None:
        print("Error: Could not find initial amount in trade_result.txt")
        exit(1)
    
    # Get final amount from last transaction or final amount line
    final_amount = initial_amount
    with open(file_path, 'r') as f:
        for line in f:
            if 'Final amount in Dollar:' in line:
                final_amount = float(re.search(r'(\d+\.\d+)', line).group(1))
                break
    
    print(f"\nInitial Capital: ${initial_amount:,.2f}")
    print(f"Final Capital: ${final_amount:,.2f}")
    print(f"Number of Transactions: {len(transactions)}")
    
    # Calculate returns
    returns = calculate_returns(initial_amount, transactions)
    
    if len(returns) == 0:
        print("\nNo complete buy/sell pairs found. Cannot calculate metrics.")
        exit(1)
    
    # Calculate equity curve
    equity_curve = calculate_equity_curve(initial_amount, transactions)
    
    # Calculate metrics
    roi = calculate_roi(initial_amount, final_amount)
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    max_dd = calculate_max_drawdown(equity_curve)
    
    # Additional statistics
    total_trades = len(returns)
    winning_trades = len([r for r in returns if r > 0])
    losing_trades = len([r for r in returns if r < 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    avg_return = np.mean(returns) * 100
    total_return = sum(returns) * 100
    
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)
    print(f"\nROI (Return on Investment): {roi:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print(f"Sortino Ratio: {sortino:.4f}")
    print(f"Maximum Drawdown: {max_dd:.2%}")
    
    print("\n" + "=" * 60)
    print("TRADE STATISTICS")
    print("=" * 60)
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Losing Trades: {losing_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Return per Trade: {avg_return:.4f}%")
    print(f"Total Return: {total_return:.2f}%")
    
    if len(returns) > 0:
        print(f"Best Trade: {max(returns) * 100:.2f}%")
        print(f"Worst Trade: {min(returns) * 100:.2f}%")
        print(f"Standard Deviation: {np.std(returns) * 100:.4f}%")
    
    print("\n" + "=" * 60)

