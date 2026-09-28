@echo off
REM ===================================================================
REM  ShipsNet k-fold training: full 252-config grid for folds 2,3,4,5.
REM  Fold 1 = existing legacy models (trained separately).
REM
REM  Per fold: 4 variants x 3 priors, each invocation sweeping
REM  7 activations x 3 prior-scale b = 21 configs. 12 invocations/fold.
REM  Epochs = 100 to match the legacy fold-1 models.
REM
REM  Resumable: train_shipsnet.py skips any (activation,prior,b) whose config
REM  JSON already exists in the target save-dir. Safe to kill and re-run this
REM  same batch after a reboot; finished configs are skipped automatically.
REM
REM  Run from repo root:  scripts\train_shipsnet_folds2to5.bat
REM ===================================================================
setlocal
set UV_LINK_MODE=copy
set PYTHONIOENCODING=utf-8
set DEPS=pyro-ppl,torch,torchvision,tqdm,scikit-learn,python-dotenv,requests,matplotlib,pandas,numpy
set EPOCH=100

for %%F in (2 3 4 5) do (
  call :do_variant %%F base ""
  call :do_variant %%F smartpool "--smartpool"
  call :do_variant %%F dropout "--dropout-mode"
  call :do_variant %%F weight_decay "--wd"
)
echo ===== ALL FOLDS 2-5 TRAINING DONE =====
goto :eof

:do_variant
set FOLD=%1
set VNAME=%2
set VFLAG=%~3
for %%P in (Gaussian_prior Laplace_prior Uniform_prior) do (
  echo === fold %FOLD%  variant %VNAME%  prior %%P ===
  uv run --link-mode=copy --with %DEPS% python scripts/train_shipsnet.py --prior %%P --epoch %EPOCH% --b-set full --num-workers 0 %VFLAG% --fold %FOLD% --save-dir results/shipsnet/bayesian/fold%FOLD%/%VNAME%
  if errorlevel 1 (
    echo FAILED: fold %FOLD% variant %VNAME% prior %%P
    exit /b 1
  )
)
goto :eof
