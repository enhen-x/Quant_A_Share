import akshare as ak
import pandas as pd
import os
import datetime
from tqdm import tqdm
import re
import time
import random

def safe_request(func, max_retries=5, sleep_min=0.5, sleep_max=1.5, **kwargs):
    """
    Akshare 接口安全调用：自动重试 + 限流
    """
    for attempt in range(max_retries):
        try:
            return func(**kwargs)
        except Exception as e:
            print(f"⚠️ 调用 Akshare 接口失败 ({func.__name__}), 重试 {attempt+1}/{max_retries} 次: {e}")
            time.sleep(random.uniform(sleep_min, sleep_max))
    print(f"❌ 最终失败：{func.__name__}")
    return None


# --- 路径配置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
# 🎯 新增目录用于存放基本面数据
FUNDAMENTAL_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw_fundamental')


def format_code(code: str) -> str:
    """将 Akshare 的纯数字代码格式化为项目代码 sh.600000 或 sz.000001"""
    if len(code) == 6:
        if code.startswith('6'):
            return f"sh.{code}"
        elif code.startswith(('0', '3')):
            return f"sz.{code}"
    return code

def get_target_stock_list():
    """
    通过 akshare 获取A股市场所有股票列表，并进行初步筛选。
    """
    print("正在通过 akshare 获取全市场股票列表...")
    
    try:
        # 使用 stock_info_a_code_name 接口获取全市场列表
        df_stocks = ak.stock_info_a_code_name()
    except Exception as e:
        print(f"❌ Akshare 获取股票列表失败: {e}")
        return []

    if df_stocks.empty:
        print("错误：获取到的股票列表为空。")
        return []
        
    df_stocks.rename(columns={'code': 'code', 'name': 'code_name'}, inplace=True)
    
    # 转换为项目格式的代码
    df_stocks['code'] = df_stocks['code'].apply(lambda x: format_code(str(x)))

    df_stocks['code'] = df_stocks['code'].astype(str)
    target_stocks = df_stocks[df_stocks['code'].str.startswith(('sh.6', 'sz.0', 'sz.3'))]['code'].tolist()
    
    # 排除科创板等
    target_stocks = [code for code in target_stocks if not code.startswith(('sh.688', 'bj', 'sz.8', 'sz.4'))]
    
    return target_stocks

def download_all_stock_history(start_date="2014-01-01"):
    """
    下载A股历史 K 线数据和最新的基本面指标 (Akshare版)。
    :param start_date: 数据起始日期 (格式: YYYY-MM-DD)
    """
    start_date_ak = start_date.replace('-', '')
    end_date_ak = datetime.datetime.now().strftime("%Y%m%d")

    # 1. 确保保存目录存在
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)
        print(f"创建 K 线数据目录: {RAW_DATA_DIR}")
        
    if not os.path.exists(FUNDAMENTAL_DATA_DIR):
        os.makedirs(FUNDAMENTAL_DATA_DIR)
        print(f"创建基本面数据目录: {FUNDAMENTAL_DATA_DIR}")

    target_stocks = get_target_stock_list()

    if not target_stocks:
        print("❌ 无法获取股票列表，下载任务终止。")
        return

    print(f"共筛选出 {len(target_stocks)} 只股票，开始下载 {start_date} 至 {end_date_ak} 的K线数据...")

    # --- 循环下载 K 线数据 (跳过已下载的文件，保持不变) ---
    skipped_count = 0
    success_count = 0
    
    for full_code in tqdm(target_stocks, desc="下载 K 线进度"):
        code = full_code.split('.')[-1]
        file_path = os.path.join(RAW_DATA_DIR, f"{full_code}.csv")
        
        # 断点续传逻辑
        if os.path.exists(file_path) and os.path.getsize(file_path) > 100:
            skipped_count += 1
            continue

        try:
            # Akshare 接口：获取前复权日 K 线数据
            df_kline = safe_request(
                ak.stock_zh_a_hist,
                symbol=code,
                period="daily",
                start_date=start_date_ak,
                end_date=end_date_ak,
                adjust="qfq"
            )


            if not df_kline.empty:
                df_kline.rename(columns={
                    '日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', 
                    '最低': 'low', '成交量': 'volume', '成交额': 'amount', 
                    '换手率': 'turn', '涨跌幅': 'pctChg',
                }, inplace=True)
                
                df_kline.insert(1, 'code', full_code)
                required_cols = ['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']
                df_kline = df_kline[[c for c in required_cols if c in df_kline.columns]]
                
                df_kline['date'] = pd.to_datetime(df_kline['date']).dt.strftime('%Y-%m-%d')
                
                df_kline.to_csv(file_path, index=False)
                success_count += 1
                
        except Exception:
            continue

    # ==========================================
    # 🎯 额外步骤：下载基本的截面基本面数据 (PE, PB, 总市值等)
    # ==========================================
    
    print("\n>>> 正在下载最新的股票基本面数据 (PE, PB, 总市值)...")
    try:
        # ✅ 最终修正接口：使用东方财富 A 股实时行情，它通常包含估值信息
        df_spot = safe_request(ak.stock_zh_a_spot_em)

        # 确定包含我们所需信息的列
        # 字段名可能为：'市盈率-动态', '市净率', '总市值'
        df_spot.rename(columns={
            '代码': 'code', 
            '市盈率-动态': 'PE', 
            '市净率': 'PB',
            '总市值': 'TotalMarketCap' # 单位：元 (需要确认单位，这里按 Akshare 常见输出)
        }, inplace=True)
        
        # 仅保留所需列
        df_fundamental = df_spot[['code', 'PE', 'PB', 'TotalMarketCap']].copy()
        
        # 补充 date 列 (假设为当前日期)
        today_date_str = datetime.datetime.now().strftime('%Y-%m-%d')
        df_fundamental['date'] = today_date_str
        
        # 格式化 code
        df_fundamental['code'] = df_fundamental['code'].apply(lambda x: format_code(str(x)))
        
        # 筛选与 K 线数据匹配的股票
        df_fundamental = df_fundamental[df_fundamental['code'].isin(target_stocks)].copy()
        
        fundamental_path = os.path.join(FUNDAMENTAL_DATA_DIR, 'latest_fundamental_indicators.csv')
        df_fundamental.to_csv(fundamental_path, index=False)
        print(f"✅ 基本面指标已下载并保存至: {fundamental_path}")
        
    except Exception as e:
        print(f"❌ 基本面数据下载失败: {e}")
        print("💡 Akshare 接口不稳定，如果 'stock_zh_a_spot_em' 仍然报错，我们可能需要暂时放弃 akshare 的基本面因子，或切换到 Baostock + Tushare 的混合方案。")
        
    print("\n" + "="*30)
    print(f"任务完成！")
    print(f"成功下载/更新 K 线数据: {success_count}")
    print(f"跳过已有: {skipped_count}")
    print(f"K 线存储位置: {RAW_DATA_DIR}")
    print("="*30)

if __name__ == "__main__":
    download_all_stock_history(start_date="2014-01-01")