# 📈 Quant_A_Share
XGBoost-based short-term alpha strategy for A-share market  
（基于 XGBoost 的 A 股短线超额收益量化策略）

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-green)](https://xgboost.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Active-success)](#)
[![License](https://img.shields.io/badge/License-MIT-black)](LICENSE)

---

## Overview
Quant_A_Share is a lightweight quantitative research and trading framework for the China A-share market. It leverages XGBoost to identify short-term alpha and implements a weekly rotation strategy targeting excess returns over a benchmark (e.g., CSI 500), with strict risk control.  
（面向个人投资者的轻量量化框架，使用 XGBoost 识别短线 Alpha，通过周度轮动在控制回撤的前提下争取战胜中证500 等基准）

Key capabilities include fully automated data pipeline, feature engineering, model training, backtesting (IS/OOS), Monte Carlo random stress tests, and live stock scanning.  
（支持全流程自动化：数据、特征、训练、回测、随机压力测试与实盘选股）

> Philosophy: Focus on favorable win-rate and payoff conditions; concentrate on the strongest candidates via “Top 3 forced allocation” during the trading week.  
> （核心理念：在胜率与赔率有利时集中持仓，实施“Top 3 强制满仓”）

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Strategy Logic](#strategy-logic)
- [Backtest Showcase](#backtest-showcase)
- [FAQ](#faq)
- [License](#license)

---

## Features
- Machine learning-driven signal generation using XGBoost classifier for 5 trading day excess return probabilities.  
  （使用 XGBoost 预测 5 日超额收益概率）
- Strict risk control:
  - Exclude ST/*ST/delisted tickers
  - Filter limit-up/limit-down untradable cases
  - Remove illiquid “zombie” stocks  
  （严格风控：剔除 ST/退市、涨跌停不可交易与低流动性标的）
- Aggressive weekly rotation with “Top 3 forced full allocation”: stay engaged in bull markets and select relative strength in sideways conditions.  
  （周度轮动 + Top 3 强制满仓，兼顾牛市参与与震荡择强）
- Comprehensive testing:
  - In-sample / out-of-sample backtests
  - Monte Carlo random backtests across 2014–2025  
  （完整回测与历史周期随机压力测试）
- Fully automated operations via a central console (`main.py`) and weekly scripts.  
  （中央控制台与周度脚本实现一键自动化）

---

## Project Structure
```text
Quant_A_Share/
├── data/                       # Local data cache (generated after run)
├── models/                     # Trained XGBoost models
├── plots/                      # Backtest charts
├── src/                        # Core modules
│   ├── data_loader.py          # Data acquisition (Baostock)
│   ├── feature_eng.py          # Feature engineering (RSI, MACD, Bollinger, etc.)
│   ├── model_trainer.py        # Model training & evaluation
│   ├── backtest.py             # Strategy backtesting
│   ├── random_backtest.py      # Monte Carlo stress test
│   ├── trader.py               # Live stock scanning
│   └── ...
├── main.py                     # Entry console
├── requirements.txt            # Dependencies
└── buy_list_YYYY-MM-DD.csv     # Daily live buy list
```

---

## Quick Start
### 1) Environment
- Python 3.8+ (Windows/macOS/Linux)

```bash
git clone https://github.com/YourUsername/Quant_A_Share.git
cd Quant_A_Share
pip install -r requirements.txt
```

### 2) Initialize Data
Run the console and choose “[1] Initialize Data (Baostock)”.

```bash
python main.py
```

Menu:
- 1: Download A-share daily data (2014–present)

### 3) Features & Training
Menu:
- 2: Feature engineering (compute indicators and labels)
- 3: Train XGBoost (outputs AUC and feature importance)

### 4) Backtest
Menu:
- 4: Strategy backtest; charts saved to `plots/`.

```bash
python main.py
# choose option 4 in the menu
```

Optional robustness test:

```bash
python src/random_backtest.py
```

### 5) Live Scanning
Menu:
- 6 or 9: Generate today’s Top 3 buy list:

```text
buy_list_YYYY-MM-DD.csv
```

---

## Strategy Logic
- Universe: Broad A-share universe; exclude low-liquidity names, STAR/NEEQ, and ST.  
  （股票池：全市场；过滤低流动性，剔除科创板/北交所/ST）
- Label: `Future_Return_10d > Benchmark_Return_10d + 3%`  
  （标签：未来5日相对基准超3%）
- Features:
  - Momentum: ROC(5/10/20)
  - Trend: MA bias (Bias)
  - Oscillators: RSI, KDJ
  - Volatility: Bollinger width (BB_Width)
  - Volume: Volume ratio (Vol_Ratio)
- Trading rules:
  - Rebalance Fri close or Mon open
  - Equal-weight Top 3 by predicted probability
  - Risk filters: skip >9.5% limit-like moves or ST constituents

---

## Backtest Showcase
![Random Backtest Stress Test](plots/random_backtest_full_history.png)

---

## FAQ
- Data source?  
  Baostock for historical daily bars. You can replace with other vendors via `src/data_loader.py`.  
  （数据源：Baostock；可在 data_loader.py 中替换）
- Benchmark?  
  CSI 500 by default; configurable in backtest module.  
  （默认中证500，可配置）
- Deployment?  
  Designed for local research; integrate with broker APIs at your own risk.  
  （偏研究使用；实盘接入需自行评估）

---

## License
MIT License. See [LICENSE](LICENSE).