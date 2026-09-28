@echo off
REM Resume the EuroSAT SEU sweep for v02_01 (smartpool) locally, picking up
REM where the Kaggle sessions left off. Completed CSVs already in
REM results/eurosat/seu/v02_01 are skipped automatically (config-level), and a
REM config interrupted mid-sweep resumes at the exact flip it stopped on.
REM
REM Run from repo root:  scripts\seu_eurosat_v02_01.bat
setlocal
set UV_LINK_MODE=copy
set PYTHONIOENCODING=utf-8
set DEPS=pyro-ppl,torch,torchvision,tqdm,scikit-learn,python-dotenv,requests,matplotlib,pandas,numpy

for %%P in (Gaussian_prior Laplace_prior Uniform_prior) do (
  echo === SEU v02_01 %%P ===
  uv run --link-mode=copy --with %DEPS% python scripts/eval_seu_eurosat.py --prior %%P --search-dir results/eurosat/bayesian/results_eurosat_v02_01 --save-dir results/eurosat/seu/v02_01 --fast-smartpool
  if errorlevel 1 ( echo FAILED v02_01 %%P & exit /b 1 )
)
echo ===== v02_01 DONE =====
