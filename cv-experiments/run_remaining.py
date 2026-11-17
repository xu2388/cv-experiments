"""
依次运行剩余实验：Exp1 → Exp2 → Exp5
每个实验的输出同时写入日志文件和终端。
"""
import subprocess, sys, os, datetime, time

BASE = r"D:\大三上的资料\假期学习记录\CV实验"
os.makedirs(os.path.join(BASE, "outputs"), exist_ok=True)
CONDA_PYTHON = r"D:\anaconda3\envs\deadiff\python.exe"

EXPERIMENTS = [
    {
        "id": 1,
        "name": "Bagging Ensemble",
        "script": "exp1_ensemble_bagging.py",
        "args": ["--batch-size", "64", "--epochs", "30", "--num-models", "3"],
    },
    {
        "id": 2,
        "name": "Semi-supervised Self-training",
        "script": "exp2_semi_self_training.py",
        "args": ["--batch-size", "64", "--epochs", "30", "--num-rounds", "3"],
    },
    {
        "id": 5,
        "name": "Continual Learning EWC",
        "script": "exp5_continual_ewc.py",
        "args": [],
    },
]


def run_experiment(exp):
    exp_id = exp["id"]
    log_file = os.path.join(BASE, "outputs", f"exp{exp_id}_log.txt")
    # Open log file in append mode with line buffering
    log_fh = open(log_file, "w", encoding="utf-8", buffering=1)

    cmd = [CONDA_PYTHON, "-u", exp["script"]] + exp["args"]

    header = (
        f"{'='*60}\n"
        f"实验 {exp_id}: {exp['name']}\n"
        f"启动时间: {datetime.datetime.now()}\n"
        f"命令: {' '.join(cmd)}\n"
        f"{'='*60}\n"
    )
    print(header, end="", flush=True)
    log_fh.write(header)
    log_fh.flush()

    proc = subprocess.Popen(
        cmd,
        cwd=BASE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,  # line buffered
        text=True,  # text mode
        encoding="utf-8",
        errors="replace",
    )

    # Read output line by line and write to both terminal and log
    for line in proc.stdout:
        print(line, end="", flush=True)
        log_fh.write(line)
        log_fh.flush()

    proc.wait()
    log_fh.close()

    if proc.returncode == 0:
        print(f"\n✅ 实验 {exp_id} 完成！返回码: {proc.returncode}\n", flush=True)
    else:
        print(f"\n⚠️  实验 {exp_id} 异常退出！返回码: {proc.returncode}\n", flush=True)

    return proc.returncode == 0


def main():
    print("=" * 60, flush=True)
    print("  剩余实验调度器", flush=True)
    print("  顺序: Exp1 (Bagging) → Exp2 (自训练) → Exp5 (EWC)", flush=True)
    print(f"  启动时间: {datetime.datetime.now()}", flush=True)
    print("=" * 60, flush=True)

    results = {}
    for exp in EXPERIMENTS:
        print(f"\n>>> 开始实验 {exp['id']}: {exp['name']}", flush=True)
        ok = run_experiment(exp)
        results[exp['id']] = "✅ 完成" if ok else "⚠️ 异常"

        if exp != EXPERIMENTS[-1]:
            print("  冷却 5 秒后启动下一个...", flush=True)
            time.sleep(5)

    print("\n\n" + "=" * 60, flush=True)
    print("  全部实验运行完毕！汇总：", flush=True)
    print("=" * 60, flush=True)
    for eid in [e["id"] for e in EXPERIMENTS]:
        print(f"  Exp{eid}: {results.get(eid, '⏭️ 跳过')}", flush=True)
    print(f"\n  完成时间: {datetime.datetime.now()}", flush=True)

    # Save summary
    import json
    summary = {
        "date": str(datetime.datetime.now()),
        "results": {str(k): v for k, v in results.items()},
    }
    with open(os.path.join(BASE, "outputs", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n汇总已保存: outputs/_summary.json", flush=True)


if __name__ == "__main__":
    main()
