# 📈 Quant_A_Share: XGBoost A-Share Short-Term Alpha Strategy
# 基于 XGBoost 的 A 股短线超额收益量化策略

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-green)](https://xgboost.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Active-success)]()

---

## 📖 Project Introduction (项目简介)

**Quant_A_Share** is a lightweight A-share quantitative trading system designed for individual investors. It uses a Machine Learning model (XGBoost) to uncover short-term Alpha signals in the market, aiming to achieve returns that surpass the benchmark index (CSI 500) via a **Weekly Rotation** strategy, while controlling drawdowns.

This project includes an end-to-end automated solution for **data acquisition, data cleaning, feature engineering, model training, strategy backtesting, robustness testing**, and **live stock scanning**.

> **Core Philosophy**: The goal is not to be right every time, but to seize opportunities with favorable win rates and payoffs by holding the top stocks (Forced Top 3) in the market during the trading week.

**Quant_A_Share** 是一个面向个人投资者的轻量级 A 股量化交易系统。它利用机器学习模型（XGBoost）挖掘市场中的短线 Alpha 信号，旨在通过**周度轮动（Weekly Rotation）**策略，在控制回撤的前提下捕捉超越基准指数（中证500）的收益。

本项目包含从**数据获取、数据清洗、特征工程、模型训练、策略回测、鲁棒性测试**到**实盘选股**的全流程自动化解决方案。

> **核心理念**：不追求每一次预测都正确，但追求在“胜率”和“赔率”有利时，满仓持有全市场最强的股票（Forced Top 3）。

---

## ✨ Key Features (核心特性)

* **🤖 Machine Learning Driven**: Uses the **XGBoost Classifier** to predict the probability of a stock generating excess return over the next 5-10 trading days.
    * 🤖 **机器学习驱动**：使用 **XGBoost Classifier** 预测股票未来 5-10 个交易日的超额收益概率。
* **🛡️ Strict Risk Control System**:
    * Automatically filters out **ST / \*ST / Delisted** risk stocks.
    * Automatically identifies and filters out **Price Limit (Limit Up/Limit Down)** stocks that are untradable.
    * Filters out low-liquidity stocks ("Zombie Stocks").
    * 🛡️ **严格风控体系**：
        * 自动剔除 **ST / \*ST / 退市** 风险股。
        * 自动识别并剔除 **涨停/跌停** 无法交易的股票。
        * 基于流动性过滤“僵尸股”。
* **🚀 Aggressive Rotation Strategy**: Adopts a **"Forced Top 3 Full Position"** logic, ensuring participation in bull markets and automatic selection of relatively strong stocks during sideways markets.
    * 🚀 **激进轮动策略**：采用 **"Top 3 强制满仓"** 逻辑，在牛市中不踏空，在震荡市中自动优选相对强势股。
* **🧪 Comprehensive Testing Framework**:
    * Supports rigorous **In-Sample / Out-of-Sample** backtesting.
    * Includes built-in **Monte Carlo Random Backtest** to simulate the strategy's robustness across the full historical cycle (2014-2025), verifying its ability to navigate various market conditions.
    * 🧪 **完整的测试框架**：
        * 支持 **In-Sample / Out-of-Sample** 严格回测。
        * 内置 **蒙特卡洛随机回测 (Random Backtest)**，模拟 2014-2025 全历史周期下的策略鲁棒性，验证穿越牛熊的能力。

    ![Random Backtest Stress Test](plots/random_backtest_full_history.png)
    * ![随机回测压力测试](plots/random_backtest_full_history.png)

* **⚙️ Fully Automated Operation**: Provides a central control console (`main.py`) and weekly automation scripts to complete the entire process from data update to stock selection with a single command.
    * ⚙️ **全自动化运维**：提供中央控制台 (`main.py`) 和周度自动化脚本，一键完成数据更新到选股的全过程。

---

## 🏗️ Project Structure (项目架构)

```text
QUANT_A_SHARE/
├── data/                       # Data Storage (Generated after local run)
├── models/                     # Trained XGBoost Models
├── plots/                      # Backtest Equity Curve Charts
├── src/                        # Core Code Library
│   ├── data_loader.py          # Data Acquisition (Baostock Source)
│   ├── feature_eng.py          # Feature Engineering (RSI, MACD, Bollinger, etc.)
│   ├── model_trainer.py        # Model Training and Evaluation
│   ├── backtest.py             # Strategy Backtesting System
│   ├── random_backtest.py      # Random Stress Testing
│   ├── trader.py               # Live Stock Scanning
│   └── ...
├── main.py                     # [Entry] Project Central Control Console
├── requirements.txt            # Dependency Libraries
└── buy_list_YYYY-MM-DD.csv     # Daily Generated Live Buy List


🚀 Quick Start (快速开始)
1. Environment Setup (环境准备)
Ensure your Python version is >= 3.8. 确保你的 Python 版本 >= 3.8。

Bash

git clone [https://github.com/YourUsername/Quant_A_Share.git](https://github.com/YourUsername/Quant_A_Share.git)
cd Quant_A_Share
pip install -r requirements.txt

2. Data Initialization (初始化数据)
Run the main program and select [1] to initialize the data (Data source is Baostock, which is free and stable). 运行主程序，选择 [1] 初始化数据（数据源为 Baostock，免费且稳定）。

Bash

python main.py
Enter 1 in the menu. The system will download A-share daily data from 2014 to the present. 在菜单中输入 1，系统将下载 2014 年至今的 A 股日线数据。

3. Model Training (训练模型)
Execute the following tasks sequentially in the menu: 在菜单中依次执行：

[2] Feature Engineering: Calculates technical indicators and generates labels.

[2] 特征工程：计算技术指标并打标签。

[3] Model Training: Trains XGBoost and outputs AUC and feature importance.

[3] 训练模型：训练 XGBoost 并输出 AUC 及特征重要性。

4. Strategy Backtesting (策略回测)
Select [4] in the menu to perform the backtest. The system will generate the equity curve chart and save it in the plots/ directory. If you want to test the strategy's stability, you can select random_backtest.py for full historical stress testing. 在菜单中选择 [4] 进行回测。系统将生成资金曲线图保存在 plots/ 目录下。 如果你想测试策略的稳定性，可以选择脚本中的 random_backtest.py 进行全历史压力测试。

5. Live Stock Scanning (实盘选股)
Select [6] or [9] in the menu. The program will output the Top 3 recommended stocks for purchase today (buy_list_xxxx-xx-xx.csv) based on the latest market data. 在菜单中选择 [6] 或 [9]。 程序会根据最新行情，输出今日推荐买入的 Top 3 股票清单 (buy_list_xxxx-xx-xx.csv)。

🧠 Strategy Logic (策略逻辑)
Component (组件),                     Definition (定义)
Stock Pool (股票池),                  "The entire market, filtered for low liquidity, excluding STAR Market/Beijing Stock Exchange/ST stocks (Top 1000 liquidity targets)."
标签 (Target),                        Future_Return_10d > Benchmark_Return_10d + 3% (Outperforming the index by 3 percentage points is classified as a positive sample).
特征 (Features),                      "Momentum: ROC (5/10/20-day return). Trend: Moving Average Bias (Bias). Oscillators: RSI, KDJ. Volatility: Bollinger Band Width (BB_Width). Volume: Volume Ratio (Vol_Ratio)."
交易规则 (Trading Rules),             "Rebalance at the end of Friday or beginning of Monday. Select the Top 3 stocks with the highest predicted probability for equal-weighted purchase. Risk Control: If a target stock has a daily increase > 9.5% (potential limit up) or includes ST, it is automatically skipped to the next candidate."

⚠️ Disclaimer (免责声明)
This project is for learning and technical exchange purposes only and does not constitute any investment advice.

Quantitative models are trained on historical data, and historical performance does not represent future results.

Live trading involves uncontrollable risks such as slippage, transaction fees, and trading halts. Users must bear all risks of capital loss themselves.

The A-share market carries huge risks; please proceed with caution.

本项目仅供学习与技术交流使用，不构成任何投资建议。

量化模型基于历史数据训练，历史业绩不代表未来表现。

实盘交易存在滑点、手续费、停牌等不可控风险，使用者需自行承担所有资金损失风险。

A 股市场风险巨大，入市需谨慎。