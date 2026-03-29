Quant Trade System v1:
An institutional-grade crypto trading research framework for developing, backtesting, and validating systematic strategies across multiple assets.

Overview:
This project provides a modular pipeline for building and evaluating quantitative trading strategies. It combines trend-following and pullback approaches within a portfolio-level simulation engine, incorporating realistic execution, risk management, and validation workflows.

Features:
Multi-strategy system (trend and pullback), portfolio-level backtesting with risk controls, dynamic position sizing and correlation filtering, technical indicators (SMA, EMA, ATR, ADX, RSI), performance analytics, and walk-forward validation across development, validation, and holdout periods.

Installation:
pip install -r requirements.txt

Usage:
python main.py --profile full_system
python run_validation.py --profile full_system

Most recent run results:
Final portfolio equity: 28508.16
Total trade events: 389

=== PERFORMANCE REPORT ===
Starting equity:        10000.0
Final equity:           28508.16
Total return (%):       185.08
Symbols traded:         6 -> ['AVAXUSDT', 'BNBUSDT', 'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']
Entries:                192
Trade lifecycles:       153
Realized events:        197
Event win rate (%):     59.9
Trade win rate (%):     48.37
Average event pnl:      94.8301
Average trade pnl:      120.9684
Average trade R:        0.5997
Trade expectancy:       120.9684
Event profit factor:    2.5014
Trade profit factor:    2.4769
Gross profit (trade):   31040.0408
Gross loss (trade):     12531.8814
Average bars/trade:     45.21
Max drawdown (%):       -9.16

=== YEARLY RETURNS ===
 year  return_pct  start_equity  end_equity
 2021       53.00      10000.00    15299.60
 2022       -0.13      15299.60    15279.68
 2023       42.27      15279.68    21738.73
 2024       28.84      21719.11    27982.13
 2025        5.48      27982.13    29515.35
 2026       -3.41      29515.35    28508.16

 === WALK-FORWARD AGGREGATES ===
windows=9, avg_return_pct=2.99, median_return_pct=-0.10, avg_max_dd_pct=-2.37, avg_pf=1.852

Issues and Limitations:
The system remains sensitive to market regime shifts, with reduced performance observed during low-trend or sideways periods, as reflected in weaker walk-forward median returns. Strategy dependence on trend-following as the primary alpha source introduces concentration risk, while pullback components contribute limited diversification. Transaction cost assumptions, slippage modelling, and liquidity constraints may not fully capture real market conditions. Additionally, parameter stability has not been exhaustively stress-tested across alternative datasets or higher-frequency regimes.

Future Improvements:
Future work should focus on improving regime detection and dynamically adjusting strategy allocation between trend and mean-reversion components. Enhancements in execution modelling, including more realistic liquidity and slippage simulation, are recommended. Incorporation of additional uncorrelated strategies, short-side logic, and cross-asset diversification may improve robustness. Further research should include extensive hyperparameter sensitivity analysis, Monte Carlo simulations, and deployment-oriented features such as live trading integration and risk monitoring systems.

