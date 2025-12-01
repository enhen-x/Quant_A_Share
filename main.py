import os
import sys
import time
import datetime  # ✅ 新增：用于获取当前日期

# --- 1. 环境路径配置 ---
# 确保项目根目录在系统路径中，以便能找到 src 模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, 'src')
sys.path.append(SRC_DIR)

# --- 2. 导入功能模块 ---
try:
    from src import data_loader
    from src import selection
    from src import feature_eng
    from src import label_maker
    from src import model_trainer
    from src import backtest
    from src import trader
    from src import audit_trades
    from src import weekly_update
except ImportError as e:
    print(f"❌ 关键模块导入失败: {e}")
    print("请确保 src/ 目录下包含所有必要的脚本文件。")
    sys.exit(1)

# --- 3. 界面辅助函数 ---
def clear_screen():
    # 简单清屏，兼容 Windows 和 Mac/Linux
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("="*50)
    print("      📈 QUANT A-SHARE (XGBoost Alpha)      ")
    print("      A股短线量化交易系统 - 中央控制台      ")
    print("="*50)

def print_menu():
    print("\n请选择要执行的任务：")
    print("-" * 30)
    print(" [1]  📥  初始化/更新数据 (下载 + 筛选)")
    print(" [2]  ⚙️  特征工程 (计算因子 + 打标签)")
    print(" [3]  🧠  训练模型 (XGBoost)")
    print(" [4]  📉  策略回测 (激进版 + 风控)")
    print(" [5]  🕵️  审计回测记录 (查ST/涨跌停)")
    print(" [6]  🚀  实盘选股 (输出今日 Buy List)")
    print("-" * 30)
    print(" [9]  🤖  一键周度更新 (自动化流水线)")
    print(" [0]  🚪  退出系统")
    print("-" * 30)

# --- 4. 任务封装 ---
def task_init_data():
    print("\n>>> 正在启动数据初始化流程...")
    # 1. 下载
    data_loader.download_all_stock_history(start_date="2014-01-01")
    # 2. 筛选
    selection.filter_stock_pool()
    input("\n✅ 数据初始化完成！按回车键返回菜单...")

def task_feature_eng():
    print("\n>>> 正在执行特征工程...")
    # 1. 计算特征
    feature_eng.process_features()
    # 2. 计算 Alpha 标签
    label_maker.make_relative_labels()
    input("\n✅ 特征工程完成！按回车键返回菜单...")

def task_train_model():
    print("\n>>> 正在启动模型训练...")
    model_trainer.train_model()
    input("\n✅ 模型训练完成！按回车键返回菜单...")

def task_backtest():
    print("\n>>> 正在启动策略回测...")
    backtest.run_backtest()
    input("\n✅ 回测完成！结果已保存在 plots/ 目录。按回车键返回...")

def task_audit():
    print("\n>>> 正在审计交易记录...")
    audit_trades.audit_backtest_trades()
    input("\n✅ 审计完成！按回车键返回...")

def task_live_trade():
    print("\n>>> 正在启动实盘扫描...")
    trader.run_scanner()
    
    # ✅ 修改点：动态获取今日日期，匹配新的文件名格式
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    print(f"\n💡 提示：请检查项目根目录下的 'buy_list_{today_str}.csv'")
    
    input("✅ 扫描完成！按回车键返回菜单...")

def task_weekly_auto():
    print("\n>>> 启动周度自动化任务...")
    weekly_update.run_weekly_routine()
    input("\n✅ 所有周度任务已执行完毕！按回车键返回...")

# --- 5. 主循环 ---
def main():
    while True:
        clear_screen()
        print_header()
        print_menu()
        
        choice = input("请输入选项序号: ").strip()
        
        if choice == '1':
            task_init_data()
        elif choice == '2':
            task_feature_eng()
        elif choice == '3':
            task_train_model()
        elif choice == '4':
            task_backtest()
        elif choice == '5':
            task_audit()
        elif choice == '6':
            task_live_trade()
        elif choice == '9':
            task_weekly_auto()
        elif choice == '0':
            print("再见！祝实盘长红！📈")
            sys.exit(0)
        else:
            input("❌ 无效选项，按回车键重试...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已强制退出。")
        sys.exit(0)