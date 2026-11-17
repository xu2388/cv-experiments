@echo off
chcp 65001 >nul
title Exp1 集成学习
cd /d "D:\大三上的资料\假期学习记录\CV实验"
python -u exp1_ensemble_bagging.py --batch-size 64 --epochs 50 --num-models 5
exit