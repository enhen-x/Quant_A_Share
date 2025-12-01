import pandas as pd
import numpy as np
import xgboost as xgb
import os
import joblib
import datetime
from tqdm import tqdm
import sys
import baostock as bs  # 引入 baostock 获取名称

# --- 引入公共特征库 ---
try:
    from src.features_lib import compute_all_features
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.features_lib import compute_all_features

# --- 路径配置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# ==========================================
# 0. 获取全市场股票名称 (用于识别 ST)
# ==========================================
def get_stock_names_map():
    """
    登录 Baostock，获取所有股票的最新名称
    返回字典: {'sh.600000': '浦发银行', ...}
    """
    print("正在联网获取最新股票名称表...")
    bs.login()
    
    name_map = {}
    
    # 尝试查询最近 5 天，只要查到数据就停止
    for i in range(5):
        date_chk = (datetime.datetime.now() - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
        rs = bs.query_all_stock(day=date_chk)
        
        data_list = []
        while rs.error_code == '0' and rs.next():
            data_list.append(rs.get_row_data())
            
        if data_list:
            df = pd.DataFrame(data_list, columns=rs.fields)
            name_map = dict(zip(df['code'], df['code_name']))
            print(f"成功获取名称表 (日期: {date_chk})，共 {len(name_map)} 只。")
            break
            
    bs.logout()
    return name_map

# ==========================================
# 1. 辅助检查函数
# ==========================================
def check_data_freshness(date_val):
    data_date = pd.to_datetime(date_val).date()
    today = datetime.datetime.now().date()
    delta = (today - data_date).days
    if delta > 3:
        return False, f"数据过期 ({data_date})"
    return True, "最新"

def is_valid_candidate(latest_row, stock_name=""):
    """
    实盘过滤器：剔除无法交易的股票
    """
    # 1. 名称检查 (核心修复：剔除 ST)
    if stock_name:
        upper_name = stock_name.upper()
        if 'ST' in upper_name:
            return False, f"ST股 ({stock_name})"
        if '退' in upper_name:
            return False, f"退市股 ({stock_name})"
            
    # 2. 停牌 (成交量为0)
    if latest_row['volume'] == 0:
        return False, "停牌"
    
    # 3. 涨停 (防止买不进)
    if latest_row['pctChg'] > 9.5:
        return False, "已涨停"
    
    # 4. 跌停
    if latest_row['pctChg'] < -9.5:
        return False, "已跌停"
    
    # 5. 价格异常
    if latest_row['close'] <= 0:
        return False, "价格异常"

    return True, "合格"

# ==========================================
# 2. 核心扫描逻辑
# ==========================================
def run_scanner():
    print("🚀 启动实盘选股扫描器 (ST 防御版)...")
    
    # 1. 准备工作
    model_path = os.path.join(MODELS_DIR, 'xgb_alpha_model.json')
    feat_path = os.path.join(MODELS_DIR, 'feature_names.pkl')
    
    if not os.path.exists(model_path):
        print("错误：未找到模型文件！")
        return

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    feature_names = joblib.load(feat_path)
    
    # 获取名称表
    name_map = get_stock_names_map()
    if not name_map:
        print("⚠️ 警告：无法获取股票名称，ST 过滤可能失效！")

    # 2. 读取股票池
    pool_path = os.path.join(PROCESSED_DIR, 'stock_pool.csv')
    stock_pool = pd.read_csv(pool_path)
    target_codes = stock_pool['code'].astype(str).tolist()
    
    scan_results = []
    
    print(f"正在扫描 {len(target_codes)} 只股票...")
    
    for code in tqdm(target_codes):
        file_path = os.path.join(RAW_DATA_DIR, f"{code}.csv")
        if not os.path.exists(file_path):
            continue
            
        try:
            df = pd.read_csv(file_path)
            if len(df) < 30: continue
            
            # 计算特征
            df = compute_all_features(df)
            latest_row = df.iloc[[-1]].copy()
            
            # 过滤器
            stock_name = name_map.get(code, "")
            valid, reason = is_valid_candidate(latest_row.iloc[0], stock_name)
            if not valid:
                continue

            if latest_row[feature_names].isnull().any().any():
                continue
                
            prob = model.predict_proba(latest_row[feature_names])[0, 1]
            
            scan_results.append({
                'code': code,
                'name': stock_name,
                'date': latest_row['date'].values[0],
                'close': latest_row['close'].values[0],
                'pctChg': latest_row['pctChg'].values[0],
                'probability': prob,
                'bb_width': latest_row['bb_width'].values[0]
            })
            
        except Exception:
            continue

    # 3. 输出 Top 3
    if scan_results:
        res_df = pd.DataFrame(scan_results)
        
        # 强制选 Top 3 (只要概率 > 0.5)
        qualified = res_df[res_df['probability'] > 0.5]
        
        if not qualified.empty:
            final_picks = qualified.sort_values(by='probability', ascending=False).head(3)
        else:
            final_picks = res_df.sort_values(by='probability', ascending=False).head(3)
        
        print("\n" + "="*70)
        print(f"🎯 最终选股结果 (已剔除 ST/涨跌停)")
        print("="*70)
        
        output_cols = ['code', 'name', 'date', 'close', 'pctChg', 'probability', 'bb_width']
        print(final_picks[output_cols].to_string(index=False))
        
        # --- ✅ 修改点：文件名加上日期 ---
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        file_name = f'buy_list_{today_str}.csv'
        save_path = os.path.join(PROJECT_ROOT, file_name)
        
        final_picks.to_csv(save_path, index=False)
        
        print("\n" + "-"*60)
        print(f"✅ 包含 ST 过滤的清单已生成: {save_path}")
        print("💡 最后一步：请务必在交易软件中再次确认 K 线形态！")
        print("-"*60)
        
    else:
        print("未扫描到有效数据。")

if __name__ == "__main__":
    run_scanner()