# CDFNN2-dx 使用说明

条件密度估计（Conditional Density Estimation）对比实验框架：在 toy 数据集和 UCI 真实数据集上训练并评估多个模型。

---

## 目录

- [模型](#模型)
- [快速开始](#快速开始)
- [run_all.py 批量运行（推荐）](#run_allpy-批量运行推荐)
- [单条命令手动运行](#单条命令手动运行)
- [训练参数](#训练参数)
- [结果位置与断点续跑](#结果位置与断点续跑)
- [目录结构](#目录结构)
- [结果收集脚本](#结果收集脚本)

---

## 模型

| 模型名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `cdf` | CDF-First（本算法） | MADE 单调 CDF 网络 `CDFValNetMadeSMN` |


> **默认核心模型**：`cdf`

---

## 快速开始
 
### 环境

- Python 3.x + PyTorch（有 GPU 优先）
- 依赖见 `requirements.txt`

### Real data（UCI 数据集）

数据循环变量是 `run_id`（每个数据集 × 10 个 run_id，train 70% / test 30%）。

```bash
# 1) 先生成数据划分
python run_all.py --preprocess
# 2) 训练（自动跳过已完成）
python run_all.py
```

### Toy data（二维条件分布）

数据在线生成、无需 preprocess，循环变量是 `seed`。

```bash
python run_all.py --toy
```

> **注意**：toy 数据是条件分布，命令里**必须带 `--condition=y`**（`run_all.py --toy` 已自动带上）。

---

## run_all.py 批量运行（推荐）

### Real data 模式

```bash
python run_all.py                       # 跑 6 个核心模型（自动跳过已完成的）
python run_all.py --dry-run             # 只打印命令，不执行
python run_all.py --models cdf
python run_all.py --datasets fish,wave  # 只跑部分数据集
python run_all.py --run_ids 0,1,2       # 只跑部分 run_id
python run_all.py --device cuda:0       # 指定 GPU
python run_all.py --split_ratio 0.7     # 改测试集比例（默认 0.3 = train 70% / test 30%）
python run_all.py --preprocess          # 先为数据集生成 10 个划分，再训练
python run_all.py --no-skip             # 已完成的也重跑
python run_all.py --collect             # 汇总所有已跑结果的 best val_nll
python run_all.py --count_params        # 统计各模型参数量（不训练）
```

### Toy 模式

```bash
python run_all.py --toy                                              # 6 模型 × 5 数据集 × 10 seed
python run_all.py --toy --models cdf --datasets squares --seeds 0,1,2
python run_all.py --toy --data_nums 500,1000,2000,5000               # 扫不同样本数
python run_all.py --toy --dry-run                                    # 只打印命令
```

### 参数说明

| 参数 | 模式 | 说明 |
| :--- | :--- | :--- |
| `--models` | 通用 | 逗号分隔模型名。默认 real/toy 均运行 6 个核心模型；可选模型需显式指定 |
| `--datasets` | 通用 | 逗号分隔数据集名。real 可选 9 个 UCI；toy 可选 5 个 |
| `--toy` | 通用 | 切到 toy 模式（不加则默认 real 模式） |
| `--run_ids` | real | 逗号分隔 run_id 列表（默认 0~9） |
| `--seeds` | toy | 逗号分隔 seed 列表（默认 0~9） |
| `--data_nums` | toy | 逗号分隔样本数 data_num（默认 1000） |
| `--split_ratio` | real | 测试集比例，默认 `0.3`（train 70% / test 30%）；改成 `0.7` 即 train 30% / test 70% |
| `--device` | 通用 | 训练设备，默认 `cuda:0` |
| `--preprocess` | real | 训练前先生成数据划分（`__data/<数据集>/split<ratio>_drop0_run*.pt`） |
| `--dry-run` | 通用 | 只打印命令，不真正执行 |
| `--no-skip` | 通用 | 已完成的任务也重跑 |
| `--collect` | real | 只汇总已有结果，不训练 |
| `--count_params` | 通用 | 只统计各模型参数量（不训练） |

### 可用的数据集

- **Real data（9 个 UCI）**：`fish` / `concrete` / `energy` / `parkinsons` / `temperature` / `air` / `skillcraft` 
- **Toy（5 个二维）**：`squares` / `four_triangle` / `elastic_ring` / `half_gauss` / `gauss_stick`

### 说明

- 多个值用**逗号**分隔，不要用空格（空格会被命令行当成另一个参数）。
- Real 模式首次运行前需要先生成数据划分：加 `--preprocess`，或手动执行
  `python -m data.from_ddn.pre_process --dataset <数据集名> --train_test_split_ratio 0.3`。
- toy 模式（`--toy`）不读数据文件、无需 preprocess：`--data_nums` 控制样本数、`--seeds` 控制随机种子。
- `hrli` 实现 Hall、Racine 与 Li（2004）《Cross-validation and the estimation of conditional probability densities》的连续变量方法：用训练集带宽标准化，并最小化留一加权最小二乘 CV 目标；多维目标使用逐维乘积高斯核。带宽在标准化空间的 `[0.03, 8]` 范围内搜索。此适配器面向项目当前的连续数值输入，未启用论文中针对离散特征的广义核。为控制高维场景的带宽搜索开销，默认从最多 128 个训练点构成的 pilot 子样本上选择带宽，再用完整训练集估计条件密度；可用 `--hrli_cv_samples` 和 `--hrli_cv_iter` 调节搜索规模。该模型一次性拟合，不使用 epoch/Adam 训练。
- real 默认任务数为 540（6 模型 × 9 数据集 × 10 run_id）；toy 默认任务数为 300（6 模型 × 5 数据集 × 10 seed）。
- `--count_params` 统计口径：神经网络模型统计 `model.parameters()`；`atm` 报告 Hermite 基系数个数；`hrli` 是非参数核估计器，因此参数量为 0。

---

## 单条命令手动运行

### Toy（cdf-first 示例）

```bash
python train/_2d/test_2d_parallel_org_space.py --net_name=cdf --data_name=squares --seed=0 \
  --split_ratio=0.5 --device=cuda:0 --condition=y --data_num=1000 \
  --total_epoch=20000 --batch_size=256 --lr=3e-4 --grad_dx=0.01 \
  --dim_condition_mul=8,8,2 --dim_feature=16,16,1 --num_group=32 --group_size=32 \
  --use_bn=t --residual=f --min_max=soft --init_logsigma=-2 \
  --vl_loss_condition=0.05 --vl_loss_point=0.01
```

### Real（cdf-first 示例）

```bash
python train/_real/real_data_test.py --net_name=cdf --data_name=fish --run_id=0 --seed=0 \
  --split_ratio=0.3 --device=cuda:0 --condition=y \
  --total_epoch=20000 --batch_size=256 --lr=3e-4 --grad_dx=0.01 \
  --dim_condition_mul=8,8,2 --dim_feature=16,16,1 --num_group=32 --group_size=32 \
  --use_bn=t --residual=f --min_max=soft --init_logsigma=-2 \
  --vl_loss_condition=0.01 --vl_loss_point=0.005
```


## 训练参数


### cdf

| 参数 | 默认 | 说明 |
| :--- | :---: | :--- |
| `--dim_condition_mul` | 1 | 条件维乘子（如 `8,8,2`） |
| `--dim_feature` | 1 | 特征维（如 `16,16,1`） |
| `--num_group` / `--group_size` | 8 / 8 | SMM 分组数与分组大小 |
| `--grad_dx` | 5e-6 | CDF 导数有限差分步长 |
| `--vl_loss_condition` / `--vl_loss_point` | 0.1 / 0.1 | 条件 / 目标加噪损失权重 |
| `--use_bn` | False | 是否用 BatchNorm |
| `--residual` | False | 是否用残差连接 |
| `--min_max` | soft | 单调网络 min/max 方式 |
| `--init_logsigma` | 0. | logσ 初始值 |



## 结果位置与断点续跑

- **Real 结果**：`exp/real_data/<data>-<split>/<net>-<args>/<seed>-<run_id>/`（日志 `log.log`、最优 `best_state.pt`）
- **Toy 结果**：`exp/test2d_parallel_exp_base/('<data>',True,<data_num>,<split>)/<seed>/<net>/<args>/`
- **断点续跑**：已完成任务记录在 `run_logs/.done.txt`（toy 在 `run_logs/toy/.done.txt`），重跑自动跳过；加 `--no-skip` 强制重跑。

---

## 目录结构

| 目录 | 说明 |
| :--- | :--- |
| `run_all.py` | 批量运行入口（real + toy） |
| `train/` | 训练脚本 |
| `train/_2d` | 2D toy 实验脚本 |
| `train/_real` | real data 实验脚本 |
| `train/train_base.py` | 训练/验证/画图公共基类、`parse_args`、`get_net` |
| `train/record` | 早停、记录 |
| `model/` | 模型基类（`base.py`、`val_made_smn.py` 等） |
| `MonoNetwork/` | 单调神经网络（SMM） |
| `data/` | 数据生成与读取脚本（`from_ddn`、`read_data_from_ddn.py`、`real_data.py`） |
| `loss/` | 损失实现 |
| `Flow/` | 流模型（`RealNVP.py`） |
| `Collect/` | 结果收集绘图脚本 |
| `__data/` | 生成的 UCI 数据划分 |
| `exp/` | 实验结果输出 |

---

## 结果收集脚本

脚本位于 `Collect/`。

### collect_real_and_toy.py

用 textual 交互式控制台，按参数/实验筛选并列出结果，导出当前结果列表的 json。

导出路径形如：

```
Collect/exports/results_'elastic_ring'_True_1000_0.5_SSE.json
```

### collect_single_plot.py

导出 toy 实验的单张图片。

```bash
python Collect/collect_single_plot.py "Collect/exports/results_'elastic_ring'_True_1000_0.5_SSE.json"
```

> json 路径记得用双引号包裹。生成路径形如 `Collect/single_plot/('elastic_ring',True,1000,0.5)/.../`

### collect_mean_std.py

导出真实实验多次运行结果的均值与标准差。

```bash
python Collect/collect_mean_std.py --data=fish --model="cdf-(-2,0.01,0.005,[16,16,1],[8,8,2],5e-06,-1.0,0.0,0.0)32x32-100110"
```

### 其他

- `collect_combine_single_plot.py` / `collect_combine_single_plot_performace.py` — 合并单图 / 性能对比图
- `collect_toy_plot.py` — toy 绘图
- `collect_log.py` — 收集日志
- `collect_oor.py` — 收集 out-of-range 统计
- `collect_mean_std_csv.py` — 导出 mean/std 为 CSV
- `collect_data_info.py` / `collect_all_data_info.py` / `collect_device_info.py` — 数据 / 设备信息收集
