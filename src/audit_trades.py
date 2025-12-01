import pandas as pd
import numpy as np
import xgboost as xgb
import os
import joblib

# --- 路径配置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw') # ✅ 需要读取原始数据
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

def audit_backtest_trades():
    print("🕵️‍♂️ 开始审计回测交易记录...")
    
    # 1. 加载数据
    data_path = os.path.join(PROCESSED_DIR, 'dataset_labeled.pkl')
    model_path = os.path.join(MODELS_DIR, 'xgb_alpha_model.json')
    feat_path = os.path.join(MODELS_DIR, 'feature_names.pkl')
    
    if not os.path.exists(data_path):
        print("错误：找不到数据集文件")
        return

    df = pd.read_pickle(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # 验证集 (最后 10%)
    split_index = int(len(df) * 0.90)
    test_df = df.iloc[split_index:].copy()
    
    # 2. 推理
    print("正在加载模型进行推理...")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    feature_names = joblib.load(feat_path)
    
    X_test = test_df[feature_names]
    test_df['pred_proba'] = model.predict_proba(X_test)[:, 1]
    
    # 3. 模拟选股并打印
    all_dates = sorted(test_df['date'].unique())
    rebalance_dates = all_dates[::5]
    
    print(f"\n{'日期':<12} | {'代码':<10} | {'预测概率':<8} | {'收盘价':<8} | {'备注'}")
    print("-" * 75)
    
    total_trades = 0
    
    for date in rebalance_dates:
        daily = test_df[test_df['date'] == date]
        if len(daily) == 0: continue
        
        # 你的策略逻辑：Top 3
        picks = daily.sort_values(by='pred_proba', ascending=False).head(3)
        
        for _, row in picks.iterrows():
            code = row['code']
            close_price = row['close']
            prob = row['pred_proba']
            
            # --- ✅ 修复核心：从原始 CSV 获取开盘价 ---
            limit_tag = ""
            raw_file_path = os.path.join(RAW_DATA_DIR, f"{code}.csv")
            
            try:
                # 为了不报错，我们去读原始文件查这一天的 Open
                # 这种方式比重跑 feature_eng 要快得多
                if os.path.exists(raw_file_path):
                    # 只读取需要的列，加速
                    raw_df = pd.read_csv(raw_file_path, usecols=['date', 'open', 'high', 'close'])
                    raw_df['date'] = pd.to_datetime(raw_df['date'])
                    
                    # 找到当天的记录
                    day_record = raw_df[raw_df['date'] == date]
                    
                    if not day_record.empty:
                        open_p = day_record.iloc[0]['open']
                        high_p = day_record.iloc[0]['high']
                        close_p = day_record.iloc[0]['close']
                        
                        # 涨停判断 1: 实体大阳线 (收盘/开盘 > 9.5%)
                        if (close_p / open_p) > 1.095:
                            limit_tag = "⚠️大阳线涨停"
                        
                        # 涨停判断 2: 一字板 (最高价=最低价=收盘价，且涨幅大)
                        # 这里简化判断：如果 High == Close 且涨幅大，可能是涨停
                        # 稍微严谨一点：如果收盘价接近 10% 或 20% 涨幅限制
                        # 这里暂只做简单的实体判断
                        
            except Exception:
                limit_tag = "数据缺失"

            # 打印
            print(f"{date.date()} | {code:<10} | {prob:.4f}   | {close_price:<8.2f} | {limit_tag}")
            total_trades += 1

    print("\n" + "="*30)
    print(f"共审计交易: {total_trades} 笔")
    print("审计建议：")
    print("1. 重点检查标有 '⚠️' 的日期。如果是‘一字板’或‘秒板’，实盘可能买不进。")
    print("2. 随机抽取 3-5 个代码，去软件上看 K 线走势，确认是否为‘妖股’。")

if __name__ == "__main__":
    audit_backtest_trades()