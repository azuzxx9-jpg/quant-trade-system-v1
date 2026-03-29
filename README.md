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
