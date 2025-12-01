import os
import sys
import time
import datetime

# --- 动态添加路径，确保能找到其他模块 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

# 引入我们之前写好的各个模块
try:
    import data_loader
    import selection
    import feature_eng
    import label_maker
    import trader
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    print("请确保 data_loader.py, selection.py, feature_eng.py 等都在 src 目录下")
    sys.exit(1)

def print_step(step_name):
    print("\n" + "="*50)
    print(f"🚀 {step_name}")
    print("="*50)

def run_weekly_routine():
    start_time = time.time()
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    print(f"开始执行周度更新任务 | 日期: {today}")

    # ==========================================
    # 第一步：全量数据更新
    # ==========================================
    print_step("Step 1: 更新全市场数据 & 指数")
    try:
        data_loader.download_all_stock_history(start_date="2014-01-01")
    except Exception as e:
        print(f"⚠️ 个股数据下载出现警告: {e}")

    try:
        label_maker.download_benchmark_index(start_date="2014-01-01")
    except Exception as e:
        print(f"⚠️ 指数下载失败: {e}")

    # ==========================================
    # 第二步：动态优选股票池
    # ==========================================
    print_step("Step 2: 重新筛选股票池 (Top 1000)")
    selection.filter_stock_pool()

    # ==========================================
    # 第三步：更新特征库 (历史训练集)
    # ==========================================
    print_step("Step 3: 更新特征工程 & 训练集")
    feature_eng.process_features()
    label_maker.make_relative_labels()

    # ==========================================
    # 第四步：实盘选股 (Inference)
    # ==========================================
    print_step("Step 4: 执行实盘选股扫描")
    trader.run_scanner()

    # ==========================================
    # 总结
    # ==========================================
    elapsed = (time.time() - start_time) / 60
    today_str = datetime.datetime.now().strftime("%Y-%m-%d") # 获取今日日期字符串
    
    print("\n" + "#"*50)
    print(f"✅ 周度任务全部完成！耗时: {elapsed:.1f} 分钟")
    print(f"请检查项目根目录下的 'buy_list_{today_str}.csv' 查看推荐股票。") # ✅ 动态显示文件名
    print("#"*50)

if __name__ == "__main__":
    print("⚠️ 警告：这将下载大量数据并重写股票池。")
    confirm = input("确认开始执行周度更新吗？(y/n): ")
    if confirm.lower() == 'y':
        run_weekly_routine()
    else:
        print("任务取消。")