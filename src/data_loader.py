import baostock as bs
import pandas as pd
import os
import datetime
from tqdm import tqdm

# --- 路径配置 ---
# 自动获取当前脚本所在的绝对路径 (src/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 向上两级找到项目根目录 (Quant_A_Share/)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
# 定义原始数据保存路径 (Quant_A_Share/data/raw)
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')

def download_all_stock_history(start_date="2014-01-01"):
    """
    下载A股历史数据 (Baostock版)
    :param start_date: 数据起始日期
    """
    # 1. 确保保存目录存在
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)
        print(f"创建数据目录: {RAW_DATA_DIR}")
    
    # 2. 登陆系统
    lg = bs.login()
    if lg.error_code != '0':
        print(f"登陆失败: {lg.error_msg}")  
        return

    print(f"正在获取全市场股票列表 (保存至: {RAW_DATA_DIR})...")
    
    # 3. 获取证券信息 (自动回溯寻找最近交易日)
    stock_list = []
    check_date = datetime.datetime.now()
    
    # 尝试回溯 10 天寻找可用数据
    for i in range(10):
        date_str = check_date.strftime("%Y-%m-%d")
        rs = bs.query_all_stock(day=date_str)
        
        temp_list = []
        while rs.error_code == '0' and rs.next():
            temp_list.append(rs.get_row_data())
            
        if len(temp_list) > 0:
            print(f"成功获取股票列表，使用日期: {date_str}")
            stock_list = temp_list
            break
        else:
            check_date = check_date - datetime.timedelta(days=1)
    
    if not stock_list:
        print("错误：过去10天都无法获取股票列表，请检查 Baostock 服务或网络。")
        bs.logout()
        return
    
    df_stocks = pd.DataFrame(stock_list, columns=rs.fields)
    
    # 4. 筛选股票
    # 筛选：只下载 沪(sh) 和 深(sz) 的股票
    df_stocks['code'] = df_stocks['code'].astype(str)
    target_stocks = df_stocks[df_stocks['code'].str.startswith(('sh.6', 'sz.0', 'sz.3'))]['code'].tolist()
    
    # 排除科创板（688）和北交所（bj/8/4）
    target_stocks = [code for code in target_stocks if not code.startswith(('sh.688', 'bj', 'sz.8', 'sz.4'))]
    
    end_date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    print(f"共筛选出 {len(target_stocks)} 只股票，开始下载...")

    # 5. 循环下载
    skipped_count = 0
    success_count = 0
    
    for code in tqdm(target_stocks, desc="下载进度"):
        file_path = os.path.join(RAW_DATA_DIR, f"{code}.csv")
        
        # --- 断点续传逻辑 ---
        # 如果文件存在且不为空，跳过
        if os.path.exists(file_path) and os.path.getsize(file_path) > 100:
            skipped_count += 1
            continue
        
        try:
            # frequency="d": 日线, adjustflag="2": 前复权
            rs = bs.query_history_k_data_plus(code,
                "date,code,open,high,low,close,volume,amount,turn,pctChg",
                start_date=start_date, 
                end_date=end_date_str,
                frequency="d", 
                adjustflag="2")
    
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            if data_list:
                result = pd.DataFrame(data_list, columns=rs.fields)
                
                # 类型转换
                numeric_cols = ['open','high','low','close','volume','amount','turn','pctChg']
                for col in numeric_cols:
                    result[col] = pd.to_numeric(result[col], errors='coerce')
                    
                result.to_csv(file_path, index=False)
                success_count += 1
                
        except Exception:
            continue

    bs.logout()
    print("\n" + "="*30)
    print(f"任务完成！")
    print(f"成功下载: {success_count}")
    print(f"跳过已有: {skipped_count}")
    print(f"存储位置: {RAW_DATA_DIR}")
    print("="*30)

if __name__ == "__main__":
    # 如果直接运行此脚本，则执行下载
    download_all_stock_history(start_date="2014-01-01")