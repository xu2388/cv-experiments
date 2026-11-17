#!/usr/bin/env python3
"""
CV 实验启动器（Windows 专用）
使用 subprocess.Popen + CREATE_NEW_CONSOLE 启动独立窗口，
完全不受 Claude Bash 会话影响。

用法：
    python launch_all.py              # 按顺序逐一启动全部 8 个实验
    python launch_all.py --parallel   # 同时启动（注意显存，最多同时 2-3 个）
    python launch_all.py 1 3 5        # 只启动指定编号的实验
"""

import subprocess, sys, os, time, datetime

BASE = r"D:\大三上的资料\假期学习记录\CV实验"
CREATE_NEW_CONSOLE = 0x00000010

EXPERIMENTS = [
    {
        "id": 1,
        "name": "集成学习 - CNN Bagging",
        "script": "exp1_ensemble_bagging.py",
        "args": ["--batch-size", "64", "--epochs", "50", "--num-models", "5"],
        "estimate": "~25min",
    },
    {
        "id": 2,
        "name": "半监督学习 - 自训练",
        "script": "exp2_semi_self_training.py",
        "args": ["--batch-size", "64", "--epochs", "30", "--num-rounds", "5"],
        "estimate": "~35min",
    },
    {
        "id": 3,
        "name": "多任务学习 - 分类+分割",
        "script": "exp3_multitask_cls_seg.py",
        "args": [],
        "estimate": "~25min",
    },
    {
        "id": 4,
        "name": "迁移学习 - 4策略对比",
        "script": "exp4_transfer_learning.py",
        "args": [],
        "estimate": "~20min",
    },
    {
        "id": 5,
        "name": "持续学习 - EWC",
        "script": "exp5_continual_ewc.py",
        "args": [],
        "estimate": "~60min",
    },
    {
        "id": 6,
        "name": "领域适应 - U-Net对抗",
        "script": "exp6_domain_adaptation_unet.py",
        "args": [],
        "estimate": "~15min",
    },
    {
        "id": 7,
        "name": "元学习 - MAML Few-shot",
        "script": "exp7_maml_fewshot.py",
        "args": [],
        "estimate": "~15min",
    },
    {
        "id": 8,
        "name": "分割增量 - 知识蒸馏",
        "script": "exp8_incremental_distillation.py",
        "args": [],
        "estimate": "~20min",
    },
]


def launch_experiment(exp):
    """Launch a single experiment in a new console window."""
    title = f"Exp{exp['id']} {exp['name']}"
    cmd = [sys.executable, "-u", exp["script"]] + exp["args"]

    proc = subprocess.Popen(
        cmd,
        cwd=BASE,
        creationflags=CREATE_NEW_CONSOLE,
        close_fds=True,
    )
    return proc


def main():
    args = sys.argv[1:]

    # 解析参数
    parallel = False
    if "--parallel" in args:
        parallel = True
        args.remove("--parallel")

    # 确定要运行的实验
    if args:
        ids = [int(a) for a in args if a.isdigit()]
        to_run = [e for e in EXPERIMENTS if e["id"] in ids]
    else:
        to_run = EXPERIMENTS

    # 按耗时排序（短 -> 长）
    time_order = {6: 0, 7: 1, 4: 2, 8: 3, 1: 4, 3: 5, 2: 6, 5: 7}
    to_run.sort(key=lambda e: time_order.get(e["id"], 99))

    print("=" * 60)
    print("CV 实验启动器")
    print("=" * 60)
    print(f"并行模式: {'是' if parallel else '否（逐个运行）'}")
    print(f"实验数量: {len(to_run)}")
    print()

    if parallel:
        print("启动所有实验...")
        for exp in to_run:
            proc = launch_experiment(exp)
            print(f"  [{exp['id']}] {exp['name']} {exp['estimate']} → PID {proc.pid}")
            time.sleep(0.5)  # 错开启动
        print("\n全部已启动！独立窗口正在运行。")
    else:
        print("逐个运行实验：")
        print()
        for i, exp in enumerate(to_run, 1):
            print("-" * 50)
            print(f"[{i}/{len(to_run)}] 实验 {exp['id']}: {exp['name']}")
            print(f"  预计耗时: {exp['estimate']}")
            print(f"  正在启动新窗口...")

            proc = launch_experiment(exp)
            print(f"  PID: {proc.pid}")
            print(f"  窗口标题: Exp{exp['id']} {exp['name']}")
            print(f"  完成后窗口自动关闭，然后启动下一个实验。")
            print()

            if i < len(to_run):
                print(f"  ⏳ 等待上一个实验完成...（不要关闭当前窗口）")
                print(f"  完成后按任意键继续，或直接关闭此窗口取消后续实验。")
                try:
                    input("  按 Enter 启动下一个实验...")
                except (EOFError, KeyboardInterrupt):
                    print("\n  已取消。")
                    return

        print("=" * 50)
        print("所有实验已完成！")

    print("\n输出目录: outputs/")
    print("各实验结果保存在 outputs/exp{id}/ 下")
    print(f"启动时间: {datetime.datetime.now()}")


if __name__ == "__main__":
    main()
