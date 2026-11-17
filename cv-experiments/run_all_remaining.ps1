# Run all remaining experiments sequentially
# Each experiment runs in the SAME terminal, one after another.
# Results are saved to outputs/exp{id}/ directories and outputs/exp{id}_log.txt files.

$BASE = "D:\大三上的资料\假期学习记录\CV实验"
$CONDA_ENV = "deadiff"
$CONDA_PYTHON = "D:\anaconda3\envs\deadiff\python.exe"

# Clean old output directories
Remove-Item -Path "$BASE\outputs\exp1" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "$BASE\outputs\exp2" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "$BASE\outputs\exp5" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "$BASE\outputs\exp7" -Recurse -Force -ErrorAction SilentlyContinue

# Clean old log files
Remove-Item -Path "$BASE\outputs\exp1_log.txt" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "$BASE\outputs\exp2_log.txt" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "$BASE\outputs\exp5_log.txt" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "$BASE\outputs\exp7_log.txt" -Force -ErrorAction SilentlyContinue

# Ensure conda is initialized
& "D:\anaconda3\shell\condabin\conda-hook.ps1" 2>$null
conda activate $CONDA_ENV 2>$null

Set-Location $BASE

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting remaining experiments..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# =============================================
# Exp 7: MAML Few-shot (uses CIFAR-100)
# Note: This runs the FULL experiment (20000 iters)
# =============================================
Write-Host ">>> [1/4] Running Exp7: MAML Few-shot ..." -ForegroundColor Yellow
$env:start1 = Get-Date
Start-Process -NoNewWindow -FilePath $CONDA_PYTHON -ArgumentList "-u", "exp7_maml_fewshot.py" -RedirectStandardOutput "$BASE\outputs\exp7_log.txt" -RedirectStandardError "$BASE\outputs\exp7_log.txt" -Wait
Write-Host "    Exp7 completed in $([math]::Round(((Get-Date) - $env:start1).TotalMinutes, 1)) min" -ForegroundColor Green
Write-Host ""

# =============================================
# Exp 1: Bagging Ensemble (uses CIFAR-10)
# =============================================
Write-Host ">>> [2/4] Running Exp1: Bagging Ensemble ..." -ForegroundColor Yellow
$env:start2 = Get-Date
Start-Process -NoNewWindow -FilePath $CONDA_PYTHON -ArgumentList "-u", "exp1_ensemble_bagging.py", "--batch-size", "64", "--epochs", "50", "--num-models", "5" -RedirectStandardOutput "$BASE\outputs\exp1_log.txt" -RedirectStandardError "$BASE\outputs\exp1_log.txt" -Wait
Write-Host "    Exp1 completed in $([math]::Round(((Get-Date) - $env:start2).TotalMinutes, 1)) min" -ForegroundColor Green
Write-Host ""

# =============================================
# Exp 2: Semi-supervised Self-training (uses CIFAR-10)
# =============================================
Write-Host ">>> [3/4] Running Exp2: Semi-supervised Self-Training ..." -ForegroundColor Yellow
$env:start3 = Get-Date
Start-Process -NoNewWindow -FilePath $CONDA_PYTHON -ArgumentList "-u", "exp2_semi_self_training.py", "--batch-size", "64", "--epochs", "30", "--num-rounds", "5" -RedirectStandardOutput "$BASE\outputs\exp2_log.txt" -RedirectStandardError "$BASE\outputs\exp2_log.txt" -Wait
Write-Host "    Exp2 completed in $([math]::Round(((Get-Date) - $env:start3).TotalMinutes, 1)) min" -ForegroundColor Green
Write-Host ""

# =============================================
# Exp 5: Continual Learning EWC (uses CIFAR-100)
# =============================================
Write-Host ">>> [4/4] Running Exp5: Continual Learning EWC ..." -ForegroundColor Yellow
$env:start4 = Get-Date
Start-Process -NoNewWindow -FilePath $CONDA_PYTHON -ArgumentList "-u", "exp5_continual_ewc.py" -RedirectStandardOutput "$BASE\outputs\exp5_log.txt" -RedirectStandardError "$BASE\outputs\exp5_log.txt" -Wait
Write-Host "    Exp5 completed in $([math]::Round(((Get-Date) - $env:start4).TotalMinutes, 1)) min" -ForegroundColor Green
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ALL EXPERIMENTS COMPLETE!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results saved to:" -ForegroundColor White
Write-Host "  Exp7: outputs\exp7\  (log: outputs\exp7_log.txt)" -ForegroundColor White
Write-Host "  Exp1: outputs\exp1\  (log: outputs\exp1_log.txt)" -ForegroundColor White
Write-Host "  Exp2: outputs\exp2\  (log: outputs\exp2_log.txt)" -ForegroundColor White
Write-Host "  Exp5: outputs\exp5\  (log: outputs\exp5_log.txt)" -ForegroundColor White
