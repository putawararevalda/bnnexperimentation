@echo off
REM Terminal A: EuroSAT SEU for base + smartpool variants (v02_00, v02_01).
REM Resumable; skips configs already done. Run: scripts\seu_eurosat_A.bat
setlocal
set UV_LINK_MODE=copy
set PYTHONIOENCODING=utf-8
set DEPS=pyro-ppl,torch,torchvision,tqdm,scikit-learn,python-dotenv,requests,matplotlib,pandas,numpy
for %%D in (v02_00 v02_01) do (
  for %%P in (Gaussian_prior Laplace_prior Uniform_prior) do (
    echo === [A] SEU %%D %%P ===
    uv run --link-mode=copy --with %DEPS% python scripts/eval_seu_eurosat.py --prior %%P --search-dir results/eurosat/bayesian/results_eurosat_%%D --save-dir results/eurosat/seu/%%D
    if errorlevel 1 ( echo FAILED %%D %%P & exit /b 1 )
  )
)
echo ===== [A] base + smartpool DONE =====
