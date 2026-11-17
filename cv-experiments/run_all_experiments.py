#!/usr/bin/env python3
"""
CV 实验总调度器 v2 — 纯 Python，在当前终端直接运行
逐个运行剩余实验，一个完成后自动启动下一个
"""
import subprocess, sys, os, time

SCRIPT_DIR = r"D:\大三上的资料\假期学习记录\CV实验"
os.makedirs(os.path.join(SCRIPT_DIR, "outputs"), exist_ok=True)

REMAINING = [
    (1, "exp1_ensemble_bagging.py", ["--batch-size", "64", "--epochs", "2", "--num-models", "2"], "~2min"),
    (3, "exp3_multitask_cls_seg.py", [], "~25min"),
    (4, "exp4_transfer_learning.py", [], "~20min"),
    (6, "exp6_domain_adaptation_unet.py", [], "~15min"),
    (7, "exp7_maml_fewshot.py", [], "~15min"),
    (8, "exp8_incremental_distillation.py", [], "~20min"),
    (2, "exp2_semi_self_training.py", ["--batch-size", "64", "--epochs", "30", "--num-rounds", "3"], "~35min"),
    (5, "exp5_continual_ewc.py", [], "~60min"),
]

for i, (eid, script, args, est) in enumerate(REMAINING, 1):
    log_file = os.path.join(SCRIPT_DIR, "outputs", f"exp{eid}_log.txt")
    print(f"\n{'='*60}")
    print(f"[{i}/{len(REMAINING)}] 实验 {eid}: {script} {est}")
    print(f"{'='*60}")
    sys.stdout.flush()

    cmd = [sys.executable, "-u", script] + args
    with open(log_file, "a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n=== Exp{eid} started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log.flush()
        proc = subprocess.Popen(
            cmd, cwd=SCRIPT_DIR,
            stdout=log, stderr=subprocess.STDOUT,
            close_fds=True,
        )
        print(f"PID: {proc.pid}  日志: outputs/exp{eid}_log.txt")
        sys.stdout.flush()
        ret = proc.wait()
    print(f"完成 (exit={ret})")

print(f"\n{'='*60}")
print("剩余实验运行完毕！")
print(f"{'='*60}")
