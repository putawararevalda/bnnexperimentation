@echo off
REM ===================================================================
REM  ShipsNet Deterministic SEU Evaluation (Folds 2 to 5)
REM
REM  Runs single-event upset evaluations across all 4 deterministic variants
REM  (00=base, 01=smartpool, 02=dropout, 03=weight_decay) and 7 activations
REM  for folds 2, 3, 4, and 5.
REM
REM  Resumable: Skips completed 84-row CSV files automatically.
REM
REM  Usage from repository root:
REM      scripts\eval_seu_deterministic_folds2to5.bat
REM ===================================================================
setlocal
set UV_LINK_MODE=copy
set PYTHONIOENCODING=utf-8
set DEPS=torch,torchvision,tqdm,scikit-learn,python-dotenv,requests,pandas,numpy

echo ===== Starting Deterministic SEU Evaluation (Folds 2-5) =====
uv run --with %DEPS% python scripts/eval_seu_deterministic_folds2to5.py %*

if errorlevel 1 (
    echo [ERROR] Deterministic SEU evaluation failed!
    exit /b 1
)

echo ===== All Deterministic Evaluations for Folds 2-5 Completed Successfully =====
