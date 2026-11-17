@echo off
chcp 65001 > nul
cd /d "D:\大三上的资料\假期学习记录\CV实验"

:: Clean old log files
if exist outputs\exp1_log.txt del /f /q outputs\exp1_log.txt
if exist outputs\exp2_log.txt del /f /q outputs\exp2_log.txt
if exist outputs\exp5_log.txt del /f /q outputs\exp5_log.txt
if exist outputs\exp7_log.txt del /f /q outputs\exp7_log.txt

echo ========================================
echo Running Exp1: Bagging Ensemble
echo ========================================
call conda activate deadiff
python -u exp1_ensemble_bagging.py --batch-size 64 --epochs 50 --num-models 5 --num-workers 0 > outputs\exp1_log.txt 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Exp1 FAILED with error code %ERRORLEVEL%
    type outputs\exp1_log.txt
) else (
    echo Exp1 completed successfully
)

echo ========================================
echo Running Exp2: Semi-Supervised Self-Training
echo ========================================
python -u exp2_semi_self_training.py --batch-size 64 --epochs 30 --num-rounds 5 --num-workers 0 > outputs\exp2_log.txt 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Exp2 FAILED with error code %ERRORLEVEL%
    type outputs\exp2_log.txt
) else (
    echo Exp2 completed successfully
)

echo ========================================
echo Running Exp5: Continual Learning EWC
echo ========================================
python -u exp5_continual_ewc.py > outputs\exp5_log.txt 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Exp5 FAILED with error code %ERRORLEVEL%
    type outputs\exp5_log.txt
) else (
    echo Exp5 completed successfully
)

echo ========================================
echo Running Exp7: MAML Few-shot
echo ========================================
python -u exp7_maml_fewshot.py > outputs\exp7_log.txt 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Exp7 FAILED with error code %ERRORLEVEL%
    type outputs\exp7_log.txt
) else (
    echo Exp7 completed successfully
)

echo ========================================
echo All experiments done!
echo ========================================
pause
