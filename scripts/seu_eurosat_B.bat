@echo off
REM Terminal B: EuroSAT SEU for dropout + weight_decay variants (v02_02, v02_03).
REM Resumable; skips configs already done. Run: scripts\seu_eurosat_B.bat
setlocal
set UV_LINK_MODE=copy
set PYTHONIOENCODING=utf-8
set DEPS=pyro-ppl,torch,torchvision,tqdm,scikit-learn,python-dotenv,requests,matplotlib,pandas,numpy
for %%D in (v02_02 v02_03) do (
  for %%P in (Gaussian_prior Laplace_prior Uniform_prior) do (
    echo === [B] SEU %%D %%P ===
    uv run --link-mode=copy --with %DEPS% python scripts/eval_seu_eurosat.py --prior %%P --search-dir results/eurosat/bayesian/results_eurosat_%%D --save-dir results/eurosat/seu/%%D
    if errorlevel 1 ( echo FAILED %%D %%P & exit /b 1 )
  )
)
echo ===== [B] dropout + weight_decay DONE =====
