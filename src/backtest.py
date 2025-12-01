import pandas as pd
import numpy as np
import xgboost as xgb
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import baostock as bs
import datetime

# --- 路径配置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')

# ==========================================
# 0. 辅助函数：获取名称 & 验证逻辑
# ==========================================
def get_stock_names_map():
    """联网获取最新股票名称表，用于剔除 ST"""
    print("正在联网获取股票名称表 (Baostock)...")
    bs.login()
    name_map = {}
    # 尝试查询最近 5 天，找到有数据的一天
    for i in range(5):
        date_chk = (datetime.datetime.now() - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
        rs = bs.query_all_stock(day=date_chk)
        data_list = []
        while rs.error_code == '0' and rs.next():
            data_list.append(rs.get_row_data())
        if data_list:
            df = pd.DataFrame(data_list, columns=rs.fields)
            name_map = dict(zip(df['code'], df['code_name']))
            break
    bs.logout()
    return name_map

def is_valid_candidate_backtest(row, stock_name=""):
    """
    回测专用过滤器：必须返回 True 才能买
    """
    # 1. 名称检查 (剔除 ST / *ST / 退市)
    if stock_name:
        upper_name = stock_name.upper()
        if 'ST' in upper_name or '退' in upper_name:
            return False
            
    # 2. 涨跌停检查 (pctChg 需要在回测中动态计算)
    # 涨幅 > 9.5% 视为涨停 (买不进)
    if row['pctChg'] > 9.5:
        return False
        
    # 跌幅 < -9.5% 视为跌停 (接飞刀风险)
    if row['pctChg'] < -9.5:
        return False
        
    return True

# ==========================================
# 1. 主回测逻辑
# ==========================================
def run_backtest():
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    print("🚀 开始回测 (激进模式: 强制 Top 3 满仓 + 严格剔除 ST/涨停)...")
    
    # 1. 加载数据
    data_path = os.path.join(PROCESSED_DIR, 'dataset_labeled.pkl')
    model_path = os.path.join(MODELS_DIR, 'xgb_alpha_model.json')
    feat_path = os.path.join(MODELS_DIR, 'feature_names.pkl')
    
    if not os.path.exists(data_path):
        print("错误：缺少数据文件！")
        return

    df = pd.read_pickle(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # 2. 补充计算 pctChg (用于过滤涨跌停)
    print("正在重算历史涨跌幅 (用于风控)...")
    df['prev_close'] = df.groupby('code')['close'].shift(1)
    df['pctChg'] = (df['close'] / df['prev_close'] - 1) * 100
    df['pctChg'] = df['pctChg'].fillna(0) 

    # 3. 划分验证集 (最后 10%)
    split_index = int(len(df) * 0.90)
    test_df = df.iloc[split_index:].copy()
    
    print(f"回测区间: {test_df['date'].min().date()} 到 {test_df['date'].max().date()}")

    # 4. 获取名称表 (用于过滤 ST)
    name_map = get_stock_names_map()

    # 5. 预计算真实收益 (T+5)
    print("正在计算每周持仓收益...")
    test_df['close_t5'] = test_df.groupby('code')['close'].shift(-5)
    test_df['real_weekly_return'] = test_df['close_t5'] / test_df['close'] - 1.0
    test_df = test_df.dropna(subset=['real_weekly_return'])

    # 6. 模型推理
    print("正在执行模型推理...")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    feature_names = joblib.load(feat_path)
    
    X_test = test_df[feature_names]
    test_df['pred_proba'] = model.predict_proba(X_test)[:, 1]

    # ==========================================
    # 7. 激进轮动循环
    # ==========================================
    all_dates = sorted(test_df['date'].unique())
    rebalance_dates = all_dates[::5] # 每周调仓
    
    strategy_capital = 1.0
    benchmark_capital = 1.0
    
    capital_curve = [1.0]
    benchmark_curve = [1.0]
    date_curve = [rebalance_dates[0]]
    
    print(f"\n开始模拟交易，共 {len(rebalance_dates)} 周...")
    
    filtered_count = 0 # 统计一共剔除了多少个无效候选
    
    for i in range(1, len(rebalance_dates)):
        curr_date = rebalance_dates[i]
        daily_snapshot = test_df[test_df['date'] == curr_date]
        
        if len(daily_snapshot) == 0: continue
        
        # --- A. 激进选股逻辑 ---
        # 1. 直接按概率从高到低排序 (无视绝对分数，只看相对排名)
        sorted_candidates = daily_snapshot.sort_values(by='pred_proba', ascending=False)
        
        picks_list = []
        
        # 2. 遍历候选列表，直到凑齐 3 个“干净”的股票
        for _, row in sorted_candidates.iterrows():
            if len(picks_list) >= 3:
                break # 凑够了，收工
            
            code = row['code']
            name = name_map.get(code, "") # 查名字
            
            # ⚠️ 关键点：这里必须通过检查才能入选
            if is_valid_candidate_backtest(row, name):
                picks_list.append(row)
            else:
                filtered_count += 1 # 记录被剔除的数量
        
        # 3. 结算
        if picks_list:
            picks = pd.DataFrame(picks_list)
            # 假设等权买入
            real_profit = picks['real_weekly_return'].mean()
            strategy_capital *= (1 + real_profit)
        else:
            # 只有全市场所有票都跌停/ST时才会走到这里
            pass
            
        # --- B. 基准结算 ---
        mkt_avg = daily_snapshot['real_weekly_return'].mean()
        benchmark_capital *= (1 + mkt_avg)
        
        # --- C. 记录 ---
        capital_curve.append(strategy_capital)
        benchmark_curve.append(benchmark_capital)
        date_curve.append(curr_date)

    # ==========================================
    # 8. 绘图
    # ==========================================
    strategy_return = (strategy_capital - 1) * 100
    benchmark_return = (benchmark_capital - 1) * 100
    alpha = strategy_return - benchmark_return
    
    print("\n" + "="*40)
    print(f"📊 激进版回测报告 (Aggressive + Filtered)")
    print("="*40)
    print(f"因风控(ST/涨跌停)剔除次数: {filtered_count}")
    print(f"策略净值: {strategy_capital:.4f} (收益率 {strategy_return:.2f}%)")
    print(f"基准净值: {benchmark_capital:.4f} (收益率 {benchmark_return:.2f}%)")
    print(f"超额收益(Alpha): {alpha:.2f}%")
    print("="*40)

    plt.figure(figsize=(12, 6))
    plt.plot(date_curve, capital_curve, color='#d62728', linewidth=2.0, label='AI Strategy (Aggressive)')
    plt.plot(date_curve, benchmark_curve, color='gray', linestyle='--', linewidth=1.5, label='Benchmark')
    
    plt.fill_between(date_curve, capital_curve, benchmark_curve, 
                     where=(np.array(capital_curve) > np.array(benchmark_curve)), 
                     facecolor='red', alpha=0.1)

    plt.title(f'Aggressive Backtest: Forced Top 3 (Alpha = {alpha:.1f}%)', fontsize=14)
    plt.ylabel('Equity Curve', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()
    
    save_path = os.path.join(PLOTS_DIR, 'final_backtest_aggressive.png')
    plt.savefig(save_path)
    print(f"📈 激进版曲线图已保存至: {save_path}")

if __name__ == "__main__":
    run_backtest()