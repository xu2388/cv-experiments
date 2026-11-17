# 计算机视觉综合实验（Exp1–Exp8）

本科阶段的计算机视觉实验合集：在 RTX 4060 Laptop（8GB 显存）上完成 8 组实验，覆盖集成学习、半监督学习、多任务学习、迁移学习、持续学习、领域适应、元学习与知识蒸馏。

## 实验列表

| 编号 | 脚本 | 实验内容 |
|------|------|----------|
| Exp1 | `exp1_ensemble_bagging.py` | Bagging 集成学习 |
| Exp2 | `exp2_semi_self_training.py` | 半监督自训练 |
| Exp3 | `exp3_multitask_cls_seg.py` | 多任务学习（分类 + 分割） |
| Exp4 | `exp4_transfer_learning.py` | 迁移学习（从零训练 / 线性探针 / 微调 / LoRA） |
| Exp5 | `exp5_continual_ewc.py` | 持续学习（EWC / 复习 / 联合训练） |
| Exp6 | `exp6_domain_adaptation_unet.py` | U-Net 对抗式领域适应 |
| Exp7 | `exp7_maml_fewshot.py` | MAML 小样本学习 |
| Exp8 | `exp8_incremental_distillation.py` | 增量蒸馏分割 |

## 怎么跑

```bash
pip install torch torchvision scikit-learn matplotlib
python exp1_ensemble_bagging.py   # 跑单个实验
python run_all_experiments.py     # 依次跑全部实验
```

Windows 下也可以直接双击 `run_exp1.bat` ~ `run_exp8.bat`。

## 目录说明

- `outputs/`：各实验的训练曲线、对比图与 `results.json`
- `outputs_exp6/`：Exp6（领域适应）的输出（原本存放在上级目录，已一并放入本仓库）

## 数据说明

实验用的图像数据（CIFAR-100 等）体积较大，未随仓库上传；脚本中保留了数据路径，运行前需按脚本内说明准备数据。
