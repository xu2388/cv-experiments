@echo off
chcp 65001 >nul
title Exp2 半监督学习
cd /d "D:\大三上的资料\假期学习记录\CV实验"
python -u exp2_semi_self_training.py --batch-size 64 --epochs 30 --num-rounds 5
exit