Mean reversion system v1

Overwrite these files:
- main.py
- run_validation.py
- src/strategy.py
- src/portfolio.py
- src/strategies/mean_reversion.py

What this version does:
- disables broken trend shorts
- keeps trend_long as the primary alpha engine
- adds mean reversion long and mean reversion short as small neutral-regime sleeves
- exits mean reversion trades quickly by RSI or ATR target
- keeps validation slicing fixed

Run:
python main.py --profile full_system
python run_validation.py --profile full_system
