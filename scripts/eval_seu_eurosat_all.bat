@echo off
REM ===================================================================
REM  EuroSAT SEU sweep: full grid, all 4 variants x 3 priors = 12 runs.
REM  Each run sweeps every config in its variant dir (63 configs),
REM  168 injections per config. Per-variant save-dirs preserve identity.
REM
REM  Resumable: eval_seu_eurosat.py skips any timestamp that already has
REM  a CSV in the save-dir. Safe to kill and re-run this same batch.
REM
REM  For a SHORTER run (b=1.0 only, ~1/3), add  --limited-mode  to the
REM  uv line below.
REM
REM  Run from repo root:  scripts\eval_seu_eurosat_all.bat
REM ===================================================================
setlocal
set UV_LINK_MODE=copy
set PYTHONIOENCODING=utf-8
set DEPS=pyro-ppl,torch,torchvision,tqdm,scikit-learn,python-dotenv,requests,matplotlib,pandas,numpy

for %%D in (v02_00 v02_01 v02_02 v02_03) do (
  for %%P in (Gaussian_prior Laplace_prior Uniform_prior) do (
    echo === SEU  variant-dir %%D  prior %%P ===
    uv run --link-mode=copy --with %DEPS% python scripts/eval_seu_eurosat.py --prior %%P --search-dir results/eurosat/bayesian/results_eurosat_%%D --save-dir results/eurosat/seu/%%D
    if errorlevel 1 (
      echo FAILED: %%D %%P
      exit /b 1
    )
  )
)
echo ===== ALL EUROSAT SEU DONE =====
