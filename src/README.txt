Institutional-grade validation framework

Files:
- run_validation.py -> place in project root
- src/validation.py -> new file

Purpose:
- separates development / validation / holdout evaluation
- adds rolling walk-forward windows
- prevents judging the system on one blended backtest only

Default split:
- development: start -> 2023-12-31
- validation: 2024-01-01 -> 2024-12-31
- holdout: 2025-01-01 -> 2026-03-25
- walk-forward: 12-month train, 3-month test, rolling after development

Run:
python run_validation.py --profile full_system
