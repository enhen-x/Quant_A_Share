import pandas as pd
import os
import datetime
from tqdm import tqdm

# --- 路径配置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

def filter_stock_pool():
    # 1. 确保输出目录存在
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    # 2. 定义硬性门槛
    CRITERIA = {
        'max_price': 25.0,          # 股价 < 25 (硬约束)
        'min_price': 3.0,           # 股价 > 3 (提高门槛，避开垃圾股)
        'min_history': 60,          # 上市 > 60天
        'active_days': 5,           # 最近5天必须有交易
        'target_pool_size': 1000    # 🎯 目标只取前1000名
    }

    print(f"正在从 {RAW_DATA_DIR} 筛选股票...")
    print(f"硬性指标: 股价 3-25元 | 目标数量: Top {CRITERIA['target_pool_size']} 流动性")

    candidates = []
    file_list = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")]
    
    # 3. 遍历初筛
    for filename in tqdm(file_list, desc="扫描中"):
        file_path = os.path.join(RAW_DATA_DIR, filename)
        
        try:
            # 读取csv (只读最后几行提速)
            # 优化：虽然读全部稳，但这里我们只关心最近的状态
            df = pd.read_csv(file_path)
            
            if len(df) < CRITERIA['min_history']: continue

            last_row = df.iloc[-1]
            code = str(last_row['code'])
            
            # --- 剔除长期停牌 ---
            last_date = pd.to_datetime(last_row['date'])
            if (datetime.datetime.now() - last_date).days > CRITERIA['active_days']:
                continue

            # --- 价格硬约束 ---
            close = last_row['close']
            if close > CRITERIA['max_price'] or close < CRITERIA['min_price']:
                continue

            # --- 排除科创板/北交所 ---
            if code.startswith(('sh.688', 'bj', 'sz.8', 'sz.4')):
                continue

            # --- 计算流动性 (最近20天平均成交额) ---
            avg_amount = df.tail(20)['amount'].mean()
            
            # 暂时先不卡死 3000万，先全部收进来，最后排座次
            candidates.append({
                'code': code,
                'name': filename.replace('.csv', ''), # 简单用文件名作名
                'close': close,
                'avg_amount': avg_amount
            })

        except Exception:
            continue

    # 4. 核心逻辑：排序与截断
    if candidates:
        df_result = pd.DataFrame(candidates)
        
        # 按【成交额】从大到小排序
        df_result = df_result.sort_values(by='avg_amount', ascending=False)
        
        # 🔪 只取前 1000 名 (或者 800)
        df_final = df_result.head(CRITERIA['target_pool_size'])
        
        output_path = os.path.join(PROCESSED_DIR, 'stock_pool.csv')
        df_final.to_csv(output_path, index=False)
        
        print("\n" + "="*30)
        print(f"筛选完成！")
        print(f"初筛合格数: {len(df_result)}")
        print(f"最终入选数: {len(df_final)} (Top {CRITERIA['target_pool_size']})")
        print(f"结果已保存: {output_path}")
        print("="*30)
        print("入选池子示例 (流动性最强):")
        print(df_final.head(5))
        print("\n入选池子示例 (流动性门槛边缘):")
        print(df_final.tail(5))
    else:
        print("无股票入选，请检查数据。")

if __name__ == "__main__":
    filter_stock_pool()