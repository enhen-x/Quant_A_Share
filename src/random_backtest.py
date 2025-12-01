import pandas as pd
import numpy as np
import xgboost as xgb
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import baostock as bs
import datetime
import random

# --- 路径配置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')

# ==========================================
# 0. 复用辅助函数
# ==========================================
def get_stock_names_map():
    print("正在联网获取股票名称表 (Baostock)...")
    bs.login()
    name_map = {}
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
    if stock_name:
        upper_name = stock_name.upper()
        if 'ST' in upper_name or '退' in upper_name: return False
    # 涨跌停过滤
    if row['pctChg'] > 9.5: return False
    if row['pctChg'] < -9.5: return False
    return True

# ==========================================
# 1. 随机回测核心逻辑 (全历史版本)
# ==========================================
def run_random_backtest(num_simulations=20, min_duration_weeks=52):
    """
    :param num_simulations: 模拟次数
    :param min_duration_weeks: 每次回测持续周数 (默认52周=1年)
    """
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    print(f"🚀 开始全历史随机回测 (2014-2025)...")
    print(f"模拟次数: {num_simulations} 次 | 每次时长 > {min_duration_weeks} 周")
    
    # --- A. 数据准备 ---
    data_path = os.path.join(PROCESSED_DIR, 'dataset_labeled.pkl')
    model_path = os.path.join(MODELS_DIR, 'xgb_alpha_model.json')
    feat_path = os.path.join(MODELS_DIR, 'feature_names.pkl')
    
    if not os.path.exists(data_path):
        print("错误：缺少数据文件！")
        return

    # 加载全量数据
    df = pd.read_pickle(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # 算历史涨跌幅 (用于风控)
    df['prev_close'] = df.groupby('code')['close'].shift(1)
    df['pctChg'] = (df['close'] / df['prev_close'] - 1) * 100
    df['pctChg'] = df['pctChg'].fillna(0)

    # ⚠️ 关键修改：不再切分验证集，使用全量数据 (df)
    full_df = df.copy()
    
    # 算真实收益 (T+5)
    full_df['close_t5'] = full_df.groupby('code')['close'].shift(-5)
    full_df['real_weekly_return'] = full_df['close_t5'] / full_df['close'] - 1.0
    full_df = full_df.dropna(subset=['real_weekly_return'])

    print(f"全历史数据范围: {full_df['date'].min().date()} 到 {full_df['date'].max().date()}")

    # 模型推理 (全量)
    print("正在对 10 年数据进行全量推理 (可能需要一点时间)...")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    feature_names = joblib.load(feat_path)
    X_test = full_df[feature_names]
    full_df['pred_proba'] = model.predict_proba(X_test)[:, 1]

    # 获取名称表
    name_map = get_stock_names_map()

    # --- B. 准备日期序列 ---
    all_dates = sorted(full_df['date'].unique())
    all_rebalance_dates = all_dates[::5] # 每周调仓点
    total_weeks = len(all_rebalance_dates)
    
    print(f"可用调仓周期: {total_weeks} 周")
    
    if total_weeks < min_duration_weeks:
        print("数据太短，无法回测。")
        return

    # --- C. 循环模拟 ---
    stats = []
    
    plt.figure(figsize=(12, 8))
    
    for sim_i in range(num_simulations):
        # 随机选择起点
        # 确保剩余时间足够 min_duration_weeks
        max_start_idx = total_weeks - min_duration_weeks
        if max_start_idx <= 0:
            start_idx = 0
        else:
            start_idx = random.randint(0, max_start_idx)
            
        # 截取一段切片
        # 这里我们设定：从随机起点开始，一直跑到数据结束，或者跑满 2 年 (100周)
        # 为了让图表整齐，建议固定回测长度，比如就跑 52 周
        end_idx = min(start_idx + min_duration_weeks, total_weeks)
        current_dates = all_rebalance_dates[start_idx : end_idx]
        
        start_date_str = current_dates[0].date()
        
        # 初始化资金
        strategy_capital = 1.0
        benchmark_capital = 1.0
        capital_curve = [1.0]
        
        print(f"模拟 {sim_i+1}/{num_simulations}: 起点 {start_date_str}...")

        # 执行回测
        for i in range(1, len(current_dates)):
            curr_date = current_dates[i]
            daily_snapshot = full_df[full_df['date'] == curr_date]
            
            if len(daily_snapshot) == 0: continue
            
            # --- 激进选股 (强制 Top 3) ---
            sorted_candidates = daily_snapshot.sort_values(by='pred_proba', ascending=False)
            picks_list = []
            
            for _, row in sorted_candidates.iterrows():
                if len(picks_list) >= 3: break
                code = row['code']
                name = name_map.get(code, "")
                if is_valid_candidate_backtest(row, name):
                    picks_list.append(row)
            
            # 结算
            if picks_list:
                real_profit = pd.DataFrame(picks_list)['real_weekly_return'].mean()
                strategy_capital *= (1 + real_profit)
            
            # 基准
            mkt_avg = daily_snapshot['real_weekly_return'].mean()
            benchmark_capital *= (1 + mkt_avg)
            
            capital_curve.append(strategy_capital)

        # 统计
        strat_ret = (strategy_capital - 1) * 100
        bench_ret = (benchmark_capital - 1) * 100
        alpha = strat_ret - bench_ret
        
        stats.append({
            'start_date': start_date_str,
            'end_date': current_dates[-1].date(),
            'strategy_ret': strat_ret,
            'benchmark_ret': bench_ret,
            'alpha': alpha
        })
        
        # 绘图 (归一化到 X 轴 0-52 周)
        plt.plot(range(len(capital_curve)), capital_curve, alpha=0.4, linewidth=1.5)

    # --- D. 汇总报告 ---
    stats_df = pd.DataFrame(stats)
    
    print("\n" + "="*60)
    print(f"📊 全历史随机回测报告 (时长固定 {min_duration_weeks} 周)")
    print("="*60)
    print(f"平均策略收益: {stats_df['strategy_ret'].mean():.2f}%")
    print(f"平均超额收益 (Alpha): {stats_df['alpha'].mean():.2f}%")
    print(f"正收益概率 (绝对): {(stats_df['strategy_ret'] > 0).mean():.2%}")
    print(f"跑赢基准概率 (相对): {(stats_df['alpha'] > 0).mean():.2%}")
    print("-" * 60)
    print(f"最差年份收益: {stats_df['strategy_ret'].min():.2f}% (开始于 {stats_df.loc[stats_df['strategy_ret'].idxmin()]['start_date']})")
    print(f"最好年份收益: {stats_df['strategy_ret'].max():.2f}% (开始于 {stats_df.loc[stats_df['strategy_ret'].idxmax()]['start_date']})")
    print("="*60)
    
    # 打印详细列表
    # print(stats_df.sort_values(by='start_date').to_string())

    # --- E. 保存图表 ---
    plt.title(f'Random 1-Year Backtest (2014-2025 Samples)', fontsize=14)
    plt.xlabel('Weeks', fontsize=12)
    plt.ylabel('Equity (Start=1.0)', fontsize=12)
    plt.grid(True, alpha=0.3)
    # 画一条 1.0 的基准线
    plt.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
    
    save_path = os.path.join(PLOTS_DIR, 'random_backtest_full_history.png')
    plt.savefig(save_path)
    print(f"📈 历史分布图已保存至: {save_path}")

if __name__ == "__main__":
    # 跑 20 次，每次固定跑 52 周 (1年)
    run_random_backtest(num_simulations=20, min_duration_weeks=52)