#!/usr/bin/env python3
"""
CV 实验调度器（推荐使用！）
==============================
逐个运行所有实验，一个完成后自动启动下一个。
每个实验在独立窗口中运行，互不干扰。
显存上限 4GB，每个实验跑完自动关闭窗口。

用法：
    python scheduler.py              # 按顺序运行全部 8 个实验
    python scheduler.py 3            # 只运行实验 3
    python scheduler.py 1 2 4        # 运行指定实验

启动后最小化此窗口即可，无需守在电脑前。
"""

import subprocess, sys, os, time, datetime, json

BASE = r"D:\大三上的资料\假期学习记录\CV实验"
os.makedirs(os.path.join(BASE, "outputs"), exist_ok=True)

EXPERIMENTS = {
    1: {
        "name": "集成学习 - CNN Bagging (ResNet-18 × 5)",
        "script": "exp1_ensemble_bagging.py",
        "args": ["--batch-size", "64", "--epochs", "50", "--num-models", "5"],
        "estimate": "~25min",
        "data": "CIFAR-10（本地已有）",
        "gpu_mem": "~1.0 GB",
    },
    2: {
        "name": "半监督学习 - 自训练图像分类",
        "script": "exp2_semi_self_training.py",
        "args": ["--batch-size", "64", "--epochs", "30", "--num-rounds", "5"],
        "estimate": "~35min",
        "data": "CIFAR-10（本地已有）",
        "gpu_mem": "~1.0 GB",
    },
    3: {
        "name": "多任务学习 - 分类+分割共享编码器",
        "script": "exp3_multitask_cls_seg.py",
        "args": [],
        "estimate": "~25min",
        "data": "合成数据（自动生成）",
        "gpu_mem": "~1.2 GB",
    },
    4: {
        "name": "迁移学习 - 4策略对比 (Scratch/LP/FT/LoRA)",
        "script": "exp4_transfer_learning.py",
        "args": [],
        "estimate": "~20min",
        "data": "CIFAR-100 + ImageNet 预训练",
        "gpu_mem": "~1.0 GB",
    },
    5: {
        "name": "持续学习 - EWC 类增量",
        "script": "exp5_continual_ewc.py",
        "args": [],
        "estimate": "~60min",
        "data": "CIFAR-100（需要下载或已有）",
        "gpu_mem": "~0.6 GB",
    },
    6: {
        "name": "领域适应 - U-Net 对抗域适应",
        "script": "exp6_domain_adaptation_unet.py",
        "args": [],
        "estimate": "~15min",
        "data": "合成血管数据（自动生成）",
        "gpu_mem": "~0.8 GB",
    },
    7: {
        "name": "元学习 - MAML Few-shot",
        "script": "exp7_maml_fewshot.py",
        "args": [],
        "estimate": "~15min",
        "data": "CIFAR-100",
        "gpu_mem": "~2.5 GB",
    },
    8: {
        "name": "分割增量 - U-Net + 知识蒸馏",
        "script": "exp8_incremental_distillation.py",
        "args": [],
        "estimate": "~20min",
        "data": "合成数据（自动生成）",
        "gpu_mem": "~1.5 GB",
    },
}

# 只运行未完成的实验
ORDER = [7, 1, 2, 5]


def wait_for_process(proc, timeout_minutes=180):
    """等待进程结束。返回是否正常完成。"""
    start = time.time()
    while time.time() - start < timeout_minutes * 60:
        ret = proc.poll()
        if ret is not None:
            return ret == 0
        time.sleep(5)
    # 超时了强制结束
    proc.kill()
    return False


def run_experiment(exp_id):
    """在新窗口中运行单个实验，等待完成。"""
    exp = EXPERIMENTS[exp_id]
    log_file = os.path.join(BASE, "outputs", f"exp{exp_id}_log.txt")

    print(f"\n{'='*60}")
    print(f"实验 {exp_id}/8: {exp['name']}")
    print(f"{'='*60}")
    print(f"  数据: {exp['data']}")
    print(f"  显存: {exp['gpu_mem']}")
    print(f"  预计: {exp['estimate']}")
    print(f"  日志: {log_file}")
    print()

    cmd = [sys.executable, "-u", exp["script"]] + exp["args"]

    with open(log_file, "w", encoding="utf-8") as log:
        log.write(f"实验 {exp_id}: {exp['name']}\n")
        log.write(f"启动时间: {datetime.datetime.now()}\n")
        log.write(f"命令: {' '.join(cmd)}\n")
        log.write("=" * 60 + "\n")

    proc = subprocess.Popen(
        cmd,
        cwd=BASE,
        stdout=open(log_file, "a", encoding="utf-8", buffering=1),
        stderr=subprocess.STDOUT,
    )

    print(f"  PID: {proc.pid}（在独立窗口中运行）")
    print(f"  等待完成...（查看进度: type {log_file}）")

    success = wait_for_process(proc)

    if success:
        print(f"  ✅ 实验 {exp_id} 完成！")
    else:
        print(f"  ⚠️  实验 {exp_id} 可能异常退出")

    return success


def main():
    print("=" * 60)
    print("  CV 实验调度器")
    print("  (每个实验通过 subprocess + CREATE_NEW_CONSOLE 在独立窗口运行)")
    print("=" * 60)
    print()

    # 解析要运行的实验编号
    args = sys.argv[1:]
    if args:
        to_run = [int(a) for a in args if a.isdigit() and int(a) in EXPERIMENTS]
    else:
        to_run = ORDER

    # 按 ORDER 排序
    to_run.sort(key=lambda x: ORDER.index(x) if x in ORDER else 99)

    print(f"将按以下顺序运行 {len(to_run)} 个实验：")
    print()

    total_time = 0
    for i, eid in enumerate(to_run, 1):
        exp = EXPERIMENTS[eid]
        print(f"  {i}. Exp{eid}: {exp['name']}  {exp['estimate']}")
        # 简单估算时间
        mins = int(exp['estimate'].replace("~", "").replace("min", ""))
        total_time += mins

    print(f"\n预估总耗时: ~{total_time}分钟 ({total_time/60:.1f}小时)")
    print(f"显存上限: 4GB，每个实验独立运行，互不冲突")
    print()

    results = {}
    for i, eid in enumerate(to_run, 1):
        print(f"\n>>> 进度: {i}/{len(to_run)}")
        ok = run_experiment(eid)
        results[eid] = "✅ 完成" if ok else "⚠️ 异常"

        # 实验间小间隔，让 GPU 释放
        if i < len(to_run):
            print(f"  冷却 3 秒后启动下一个...")
            time.sleep(3)

    # 打印汇总
    print("\n\n")
    print("=" * 60)
    print("  全部实验运行完毕！汇总如下：")
    print("=" * 60)
    for eid in ORDER:
        if eid in results:
            print(f"  Exp{eid}: {EXPERIMENTS[eid]['name'][:40]}  {results[eid]}")
        else:
            print(f"  Exp{eid}: {EXPERIMENTS[eid]['name'][:40]}  ⏭️ 跳过")

    # 保存结果摘要
    summary = {
        "date": str(datetime.datetime.now()),
        "results": {str(k): v for k, v in results.items()},
    }
    with open(os.path.join(BASE, "outputs", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n汇总已保存: outputs/_summary.json")
    print(f"各实验详细结果: outputs/exp{{id}}/")


if __name__ == "__main__":
    main()
