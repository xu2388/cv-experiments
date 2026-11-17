@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo   CV 实验批量运行器
echo   一键运行全部 8 个实验
echo ========================================
echo.
echo 开始时间: %date% %time%
echo.

set BASE_DIR=D:\大三上的资料\假期学习记录\CV实验
set LOG_DIR=%BASE_DIR%\logs

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

:: ─── 实验 1：集成学习 ───
echo [1/8] 集成学习 - CNN Bagging...
start "Exp1 集成学习" /MIN cmd /c ""%BASE_DIR%\run_exp1.bat""

:: ─── 实验 4：迁移学习 ───
echo [2/8] 迁移学习 - 4策略对比...
start "Exp4 迁移学习" /MIN cmd /c ""%BASE_DIR%\run_exp4.bat""

:: ─── 实验 6：领域适应 ───
echo [3/8] 领域适应 - U-Net对抗...
start "Exp6 领域适应" /MIN cmd /c ""%BASE_DIR%\run_exp6.bat""

:: ─── 实验 7：MAML元学习 ───
echo [4/8] MAML元学习 - Few-shot...
start "Exp7 MAML" /MIN cmd /c ""%BASE_DIR%\run_exp7.bat""

:: ─── 实验 8：分割蒸馏 ───
echo [5/8] 分割增量 - 知识蒸馏...
start "Exp8 分割蒸馏" /MIN cmd /c ""%BASE_DIR%\run_exp8.bat""

:: ─── 实验 2：半监督 ───
echo [6/8] 半监督学习 - 自训练...
start "Exp2 半监督" /MIN cmd /c ""%BASE_DIR%\run_exp2.bat""

:: ─── 实验 3：多任务学习 ───
echo [7/8] 多任务学习 - 分类+分割...
start "Exp3 多任务" /MIN cmd /c ""%BASE_DIR%\run_exp3.bat""

:: ─── 实验 5：持续学习EWC ───
echo [8/8] 持续学习 - EWC...
start "Exp5 EWC" /MIN cmd /c ""%BASE_DIR%\run_exp5.bat""

echo.
echo 所有实验已启动！每个实验在独立的最小化窗口中运行。
echo 日志将保存到: %LOG_DIR%
echo.
echo 实验完成时间（预计）：
echo   集成学习 ~25min
echo   迁移学习 ~20min
echo   领域适应 ~15min
echo   MAML     ~15min
echo   分割蒸馏 ~20min
echo   半监督   ~35min
echo   多任务   ~25min
echo   持续学习 ~60min
echo.
echo 完成后窗口会自动关闭。
echo 按任意键关闭此窗口...
pause >nul
