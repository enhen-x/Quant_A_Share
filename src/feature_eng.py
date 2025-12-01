import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# --- 路径配置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

# ==========================================
# 1. 技术指标计算函数 (纯 Pandas 实现)
# ==========================================

def cal_rsi(series, periods=14):
    """相对强弱指标 (RSI)"""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    
    # 使用 Wilder 的平滑移动平均 (alpha = 1/n)
    ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    
    rsi = ma_up / (ma_up + ma_down) * 100
    return rsi

def cal_macd(close, fast_period=12, slow_period=26, signal_period=9):
    """移动平均聚散指标 (MACD)"""
    # 快速线 EMA
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    # 慢速线 EMA
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    # DIF
    dif = ema_fast - ema_slow
    # DEA (Signal Line)
    dea = dif.ewm(span=signal_period, adjust=False).mean()
    # MACD Histogram (柱状图) * 2 通常是为了放大显示，这里用原始值
    macd_hist = (dif - dea)
    return dif, dea, macd_hist

def cal_kdj(high, low, close, n=9, m1=3, m2=3):
    """随机指标 (KDJ)"""
    # 这里的 min_periods=1 保证初期有数据，n天内的最低低点
    low_list = low.rolling(window=n, min_periods=1).min()
    # n天内的最高高点
    high_list = high.rolling(window=n, min_periods=1).max()
    
    # RSV = (Close - Lowest_n) / (Highest_n - Lowest_n) * 100
    rsv = (close - low_list) / (high_list - low_list) * 100
    # 处理除以0的情况
    rsv = rsv.fillna(0)
    
    # 计算 K, D, J
    # K = 2/3 * Prev_K + 1/3 * RSV
    # 迭代计算比较慢，使用 pandas 的 ewm 近似模拟 (alpha=1/3)
    k = rsv.ewm(alpha=1/m1, adjust=False).mean()
    d = k.ewm(alpha=1/m2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j

def cal_bollinger(close, window=20, num_std=2):
    """布林带 (Bollinger Bands)"""
    rolling_mean = close.rolling(window=window).mean()
    rolling_std = close.rolling(window=window).std()
    
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    
    # 布林带宽度 (Band Width) - 衡量波动率
    # 处理分母为0的情况 (极少见)
    bw = (upper_band - lower_band) / rolling_mean
    return bw, (close - rolling_mean) / rolling_std # 同时也返回 Z-score

# ==========================================
# 2. 主处理逻辑
# ==========================================

def process_features():
    # 1. 读取筛选后的股票池
    pool_path = os.path.join(PROCESSED_DIR, 'stock_pool.csv')
    if not os.path.exists(pool_path):
        print("错误：未找到 stock_pool.csv，请先运行 selection.py")
        return

    stock_pool = pd.read_csv(pool_path)
    # 确保 code 是字符串
    target_codes = stock_pool['code'].astype(str).tolist()
    
    print(f"开始处理特征工程，目标股票数: {len(target_codes)}")
    
    all_data = []

    # 2. 遍历每只股票
    for code in tqdm(target_codes, desc="构造特征"):
        file_path = os.path.join(RAW_DATA_DIR, f"{code}.csv")
        if not os.path.exists(file_path):
            continue
            
        try:
            df = pd.read_csv(file_path)
            
            # 确保按时间排序
            df = df.sort_values('date').reset_index(drop=True)
            
            # --- A. 基础清洗 ---
            # 只要需要的列
            cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pctChg']
            # 有时候 raw data 可能缺某些列，做个交集
            cols = [c for c in cols if c in df.columns]
            df = df[cols].copy()

            # --- B. 构造特征 (Feature Engineering) ---
            
            # 1. 动量类 (Momentum)
            df['roc_5'] = df['close'].pct_change(5)   # 5日涨跌幅
            df['roc_10'] = df['close'].pct_change(10) # 10日涨跌幅
            df['roc_20'] = df['close'].pct_change(20) # 20日涨跌幅
            
            # 2. 均线趋势 (MA Trend)
            df['ma5'] = df['close'].rolling(5).mean()
            df['ma10'] = df['close'].rolling(10).mean()
            df['ma20'] = df['close'].rolling(20).mean()
            # 均线乖离率 (Bias)
            df['bias_20'] = (df['close'] - df['ma20']) / df['ma20']
            
            # 3. 情绪类 (Oscillators)
            # RSI
            df['rsi_6'] = cal_rsi(df['close'], 6)
            df['rsi_12'] = cal_rsi(df['close'], 12)
            df['rsi_gap'] = df['rsi_6'] - df['rsi_12'] # 短线情绪 - 长线情绪
            
            # MACD
            df['dif'], df['dea'], df['macd_hist'] = cal_macd(df['close'])
            
            # KDJ
            df['kdj_k'], df['kdj_d'], df['kdj_j'] = cal_kdj(df['high'], df['low'], df['close'])
            
            # 4. 波动率 (Volatility)
            df['bb_width'], df['bb_zscore'] = cal_bollinger(df['close'])
            
            # 5. 量能 (Volume)
            # 量比: 今日成交量 / 过去5日均量
            df['vol_ma5'] = df['volume'].rolling(5).mean()
            df['vol_ratio'] = df['volume'] / df['vol_ma5']
            
            # --- C. 构造标签 (Label Generation) ---
            # 目标: 预测未来 10 个交易日后的收益率
            # shift(-10) 表示把 10 天后的 close 移到现在这一行
            HOLDING_PERIOD = 10
            TARGET_PCT = 0.05 # 5%
            
            df['future_close'] = df['close'].shift(-HOLDING_PERIOD)
            df['future_return'] = df['future_close'] / df['close'] - 1.0
            
            # Label = 1 如果未来涨幅 > 5%，否则 0
            df['target'] = (df['future_return'] > TARGET_PCT).astype(int)
            
            # --- D. 数据清洗 (Drop NaNs) ---
            # 1. 去除前面计算指标产生的 NaN (比如 MA20 导致前19天为空)
            # 2. 去除后面因为 shift(-10) 产生的 NaN (最后10天没有未来数据)
            df = df.dropna()
            
            # 添加 code 列用于区分
            df['code'] = code
            
            # 瘦身：只保留 date, code, target 和 特征列
            # 训练不需要 open/high/low/amount，除非你用它们做特征
            # 这里保留 close 方便后续回测计算收益
            feature_cols = [
                'code', 'date', 'close', 
                'roc_5', 'roc_10', 'roc_20',
                'bias_20',
                'rsi_6', 'rsi_12', 'rsi_gap',
                'dif', 'dea', 'macd_hist',
                'kdj_k', 'kdj_d', 'kdj_j',
                'bb_width', 'bb_zscore',
                'vol_ratio',
                'target',  # <--- 这是我们要预测的
                'future_return' # <--- ✅ 必须加上这一行！
            ]
            
            all_data.append(df[feature_cols])
            
        except Exception as e:
            # print(f"Error processing {code}: {e}")
            continue

    # 3. 合并并保存
    if all_data:
        print("正在合并数据集...")
        final_df = pd.concat(all_data, ignore_index=True)
        
        # 优化内存：转为 float32
        float_cols = final_df.select_dtypes(include=['float64']).columns
        final_df[float_cols] = final_df[float_cols].astype('float32')
        
        # 保存为 Pickle 格式 (比 CSV 快且保留类型)
        output_path = os.path.join(PROCESSED_DIR, 'dataset_labeled.pkl')
        final_df.to_pickle(output_path)
        
        # 另外存一份 CSV 方便你用 Excel 查看 (只存前 1000 行示例)
        sample_path = os.path.join(PROCESSED_DIR, 'dataset_sample.csv')
        final_df.head(1000).to_csv(sample_path, index=False)
        
        print("\n" + "="*30)
        print(f"特征工程完成！")
        print(f"总样本量: {len(final_df)} 行")
        print(f"特征列: {len(feature_cols) - 3} 个") # 减去 code, date, target
        print(f"正样本(上涨)比例: {final_df['target'].mean():.2%}")
        print(f"数据已保存至: {output_path}")
        print("="*30)
    else:
        print("错误：未能生成任何数据，请检查原始数据。")

if __name__ == "__main__":
    process_features()