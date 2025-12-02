import baostock as bs
import pandas as pd
import numpy as np
import os
import datetime

# --- 路径配置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

# ==========================================
# 1. 下载基准指数数据 (Benchmark)
# ==========================================
def download_benchmark_index(start_date="2014-01-01"): 
    """
    下载中证500指数 (sh.000905) 作为基准
    """
    index_file = os.path.join(RAW_DATA_DIR, 'benchmark_sh000905.csv')
    
    # 强制重新下载：为了防止旧的时间不对的文件干扰，这里先尝试删除旧文件
    if os.path.exists(index_file):
        try:
            # 读取旧文件检查一下日期，或者直接删了重下比较稳妥
            os.remove(index_file)
            print(f"已删除旧的指数文件，准备重新下载: {index_file}")
        except Exception:
            pass

    print(f"正在下载中证500指数数据 ({start_date} 至今)...")
    
    # 登陆 Baostock
    lg = bs.login()
    if lg.error_code != '0':
        print(f"Baostock 登陆失败: {lg.error_msg}")
        return None
    
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # 获取指数数据
    rs = bs.query_history_k_data_plus("sh.000905",
        "date,close",
        start_date=start_date, end_date=end_date, frequency="d")
    
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
        
    bs.logout()
    
    if data_list:
        df_index = pd.DataFrame(data_list, columns=['date', 'close'])
        df_index['close'] = df_index['close'].astype(float)
        # 保存下来
        df_index.to_csv(index_file, index=False)
        print("基准指数下载完成。")
        return df_index
    else:
        raise ValueError("无法下载指数数据，请检查网络！")

# ==========================================
# 2. 核心逻辑：重新打标签 (Relabeling)
# ==========================================
def make_relative_labels():
    # A. 加载之前 feature_eng 生成的数据集
    dataset_path = os.path.join(PROCESSED_DIR, 'dataset_labeled.pkl')
    if not os.path.exists(dataset_path):
        print("错误：未找到 dataset_labeled.pkl，请先运行 feature_eng.py")
        return

    print("1. 读取现有数据集...")
    df = pd.read_pickle(dataset_path)
    print(f"   原始样本量: {len(df)}")

    # B. 获取基准指数数据 (会自动下载 2014-01-01 开始的数据)
    df_index = download_benchmark_index(start_date="2014-01-01")
    
    if df_index is None:
        return

    # C. 计算指数的未来收益率
    HOLDING_PERIOD = 5
    
    print("2. 计算基准指数同期涨幅...")
    df_index = df_index.sort_values('date')
    df_index['bench_close_future'] = df_index['close'].shift(-HOLDING_PERIOD)
    df_index['bench_return'] = df_index['bench_close_future'] / df_index['close'] - 1.0
    
    # 只保留 date 和 bench_return
    df_index_clean = df_index[['date', 'bench_return']].dropna()

    # D. 将指数收益率合并到个股数据中
    print("3. 合并个股与指数数据...")
    df['date'] = df['date'].astype(str)
    df_index_clean['date'] = df_index_clean['date'].astype(str)
    
    # Left Join
    df_merged = pd.merge(df, df_index_clean, on='date', how='left')

    # E. 重新定义 Target
    print("4. 重新计算 Alpha 标签...")
    
    # 填充空值：如果某天没有指数数据，假设指数涨幅为0 (保持个股原样)
    df_merged['bench_return'] = df_merged['bench_return'].fillna(0)
    
    # 计算超额收益
    df_merged['excess_return'] = df_merged['future_return'] - df_merged['bench_return']
    
    # --- 标签定义 ---
    # 目标：跑赢中证500指数 3% 以上
    ALPHA_THRESHOLD = 0.03

    # 【新增约束】：目标 2：个股未来 5 日绝对收益必须大于 1% (确保大致向上)
    MIN_ABS_RETURN_THRESHOLD = 0.01
    # 最终 Target 定义：必须同时满足超额收益和绝对收益双重条件
    df_merged['target'] = (
        (df_merged['excess_return'] > ALPHA_THRESHOLD) & 
        (df_merged['future_return'] > MIN_ABS_RETURN_THRESHOLD)
    ).astype(int)

    # 统计对比
    old_pos_ratio = df['target'].mean()
    new_pos_ratio = df_merged['target'].mean()
    
    print("\n" + "-"*30)
    print("标签修正对比:")
    print(f"原标准 (绝对涨幅 > 5%): 正样本比例 {old_pos_ratio:.2%}")
    print(f"新标准 (跑赢指数 > 3%): 正样本比例 {new_pos_ratio:.2%}")
    print("-"*30)
    
    # F. 保存
    del df_merged['bench_return'] # 删除辅助列
    
    output_path = os.path.join(PROCESSED_DIR, 'dataset_labeled.pkl')
    df_merged.to_pickle(output_path)
    print(f"5. 新数据集已保存至: {output_path}")

if __name__ == "__main__":
    make_relative_labels()