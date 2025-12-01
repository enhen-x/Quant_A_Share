# src/features_lib.py
import pandas as pd
import numpy as np

def cal_rsi(series, periods=6): # 统一默认值
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    return ma_up / (ma_up + ma_down) * 100

def cal_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd_hist = (dif - dea)
    return dif, dea, macd_hist

def cal_kdj(high, low, close, n=9, m1=3, m2=3):
    low_list = low.rolling(window=n, min_periods=1).min()
    high_list = high.rolling(window=n, min_periods=1).max()
    rsv = (close - low_list) / (high_list - low_list) * 100
    rsv = rsv.fillna(0)
    k = rsv.ewm(alpha=1/m1, adjust=False).mean()
    d = k.ewm(alpha=1/m2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j

def cal_bollinger(close, window=20, num_std=2):
    rolling_mean = close.rolling(window=window).mean()
    rolling_std = close.rolling(window=window).std()
    upper = rolling_mean + (rolling_std * num_std)
    lower = rolling_mean - (rolling_std * num_std)
    bw = (upper - lower) / rolling_mean
    zscore = (close - rolling_mean) / rolling_std
    return bw, zscore

def compute_all_features(df):
    """
    统一的特征计算入口，训练和实盘只调用这一个函数！
    """
    df = df.copy()
    # 动量
    df['roc_5'] = df['close'].pct_change(5)
    df['roc_10'] = df['close'].pct_change(10)
    df['roc_20'] = df['close'].pct_change(20)
    
    # 均线
    df['ma20'] = df['close'].rolling(20).mean()
    df['bias_20'] = (df['close'] - df['ma20']) / df['ma20']
    
    # 指标
    df['rsi_6'] = cal_rsi(df['close'], 6)
    df['rsi_12'] = cal_rsi(df['close'], 12)
    df['rsi_gap'] = df['rsi_6'] - df['rsi_12']
    
    df['dif'], df['dea'], df['macd_hist'] = cal_macd(df['close'])
    df['kdj_k'], df['kdj_d'], df['kdj_j'] = cal_kdj(df['high'], df['low'], df['close'])
    
    df['bb_width'], df['bb_zscore'] = cal_bollinger(df['close'])
    
    # 量能
    df['vol_ma5'] = df['volume'].rolling(5).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma5']
    
    return df