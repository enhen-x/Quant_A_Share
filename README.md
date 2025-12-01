# 📈 Quant_A_Share: XGBoost A-Share Short-Term Alpha Strategy
# 基于 XGBoost 的 A 股短线超额收益量化策略

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-green)](https://xgboost.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Active-success)]()

## 📖 项目简介 (Introduction)

**Quant_A_Share** 是一个面向个人投资者的轻量级 A 股量化交易系统。它利用机器学习模型（XGBoost）挖掘市场中的短线 Alpha 信号，旨在通过**周度轮动（Weekly Rotation）**策略，在控制回撤的前提下捕捉超越基准指数（中证500）的收益。

本项目包含从**数据获取、数据清洗、特征工程、模型训练、策略回测、鲁棒性测试**到**实盘选股**的全流程自动化解决方案。

> **核心理念**：不追求每一次预测都正确，但追求在“胜率”和“赔率”有利时，满仓持有全市场最强的股票（Forced Top 3）。

---

## ✨ 核心特性 (Key Features)

* **🤖 机器学习驱动**：使用 **XGBoost Classifier** 预测股票未来 5-10 个交易日的超额收益概率。
* **🛡️ 严格风控体系**：
    * 自动剔除 **ST / *ST / 退市** 风险股。
    * 自动识别并剔除 **涨停/跌停** 无法交易的股票。
    * 基于流动性过滤“僵尸股”。
* **🚀 激进轮动策略**：采用 **"Top 3 强制满仓"** 逻辑，在牛市中不踏空，在震荡市中自动优选相对强势股。
* **🧪 完整的测试框架**：
    * 支持 **In-Sample / Out-of-Sample** 严格回测。
    * 内置 **蒙特卡洛随机回测 (Random Backtest)**，模拟 2014-2025 全历史周期下的策略鲁棒性，验证穿越牛熊的能力。
* **⚙️ 全自动化运维**：提供中央控制台 (`main.py`) 和周度自动化脚本，一键完成数据更新到选股的全过程。

---

## 🏗️ 项目架构 (Structure)

```text
QUANT_A_SHARE/
├── data/                       # 数据存储 (本地运行后生成)
├── models/                     # 训练好的 XGBoost 模型
├── plots/                      # 回测资金曲线图
├── src/                        # 核心代码库
│   ├── data_loader.py          # 数据下载 (Baostock源)
│   ├── feature_eng.py          # 特征工程 (RSI, MACD, Bollinger等)
│   ├── model_trainer.py        # 模型训练与评估
│   ├── backtest.py             # 策略回测系统
│   ├── random_backtest.py      # 随机压力测试
│   ├── trader.py               # 实盘选股推演
│   └── ...
├── main.py                     # [入口] 项目中央控制台
├── requirements.txt            # 依赖库
└── buy_list_YYYY-MM-DD.csv     # 每日生成的实盘买入清单


🚀 快速开始 (Quick Start)
1. 环境准备
确保你的 Python 版本 >= 3.8。

Bash

git clone [https://github.com/YourUsername/Quant_A_Share.git](https://github.com/YourUsername/Quant_A_Share.git)
cd Quant_A_Share
pip install -r requirements.txt
2. 初始化数据
运行主程序，选择 [1] 初始化数据（数据源为 Baostock，免费且稳定）。

Bash

python main.py
在菜单中输入 1，系统将下载 2014 年至今的 A 股日线数据。

3. 训练模型
在菜单中依次执行：

[2] 特征工程：计算技术指标并打标签。

[3] 训练模型：训练 XGBoost 并输出 AUC 及特征重要性。

4. 策略回测
在菜单中选择 [4] 进行回测。系统将生成资金曲线图保存在 plots/ 目录下。 如果你想测试策略的稳定性，可以选择脚本中的 random_backtest.py 进行全历史压力测试。

5. 实盘选股
在菜单中选择 [6] 或 [9]。 程序会根据最新行情，输出今日推荐买入的 Top 3 股票清单 (buy_list_xxxx-xx-xx.csv)。

🧠 策略逻辑 (Strategy Logic)
股票池 (Universe)：全市场剔除科创板/北交所，剔除 ST，剔除低流动性个股，精选 Top 1000 流动性标的。

标签 (Target)：Future_Return_10d > Benchmark_Return_10d + 3% (跑赢指数 3 个点为正样本)。

特征 (Features)：

动量类：ROC (5/10/20日涨跌幅)

趋势类：均线乖离率 (Bias)

情绪类：RSI, KDJ

波动率：布林带宽度 (BB_Width)

量能：量比 (Vol_Ratio)

交易规则：

每周五尾盘或周一早盘调仓。

选取预测概率最高的 3 只 股票等权买入。

风控：若目标股票当日 涨幅 > 9.5% (疑似涨停) 或包含 ST，则自动顺延至下一名。

⚠️ 免责声明 (Disclaimer)
本项目仅供学习与技术交流使用，不构成任何投资建议。

量化模型基于历史数据训练，历史业绩不代表未来表现。

实盘交易存在滑点、手续费、停牌等不可控风险，使用者需自行承担所有资金损失风险。

A 股市场风险巨大，入市需谨慎。