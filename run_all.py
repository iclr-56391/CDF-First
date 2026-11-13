#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量运行条件密度估计模型，覆盖 9 个 UCI 数据集 × 10 个 run_id（train 70% / test 30%）。


每个模型自己的超参数都显式写在 MODEL_ARGS 里（也会原样拼进命令行），
要调参直接改下面的函数即可。结果保存在：
    exp/real_data/<data>-0.3/<net>-<args>/<seed>-<run_id>/log.log

用法（在项目根目录、py310 conda 环境内）：
    python run_all.py                       # 跑 6 个核心模型（自动跳过已完成的）
    python run_all.py --dry-run             # 只打印命令，不执行
    python run_all.py --models cdf     # 只跑部分模型
    python run_all.py --datasets fish,energy
    python run_all.py --run_ids 0,1,2
    python run_all.py --device cuda:0
    python run_all.py --split_ratio 0.7     # 改测试集比例（默认 0.3 = train 70% / test 30%）
    python run_all.py --preprocess          # 先为所有数据集生成 10 个划分，再训练
    python run_all.py --no-skip             # 已完成的也重跑
    python run_all.py --collect             # 汇总所有已跑结果的 best val_nll

toy 实验（train/_2d，数据在线生成，循环变量是 seed）：
    python run_all.py --toy                 # 6 个核心模型 × 5 个 toy 数据集 × 10 seed
    python run_all.py --toy --models cdf --datasets squares,elastic_ring --seeds 0,1,2
    python run_all.py --toy --data_nums 500,1000,2000,5000   # 扫不同样本数

首次运行前，需要先为每个数据集生成 10 个随机划分（等价于 --preprocess）：
    for d in fish concrete energy parkinsons temperature air skillcraft; do
        python -m data.from_ddn.pre_process --dataset $d --train_test_split_ratio 0.3
    done

提示：real 默认共 540 个任务（6 模型 × 9 数据集 × 10 run_id）；toy 默认共 300 个任务。
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ENTRY = os.path.join(PROJECT_ROOT, "train", "_real", "real_data_test.py")

# ---------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------
DEVICE = "cuda:0"      # 训练设备（可用 --device 覆盖）
SEED = 0               # 模型随机种子（固定；10 个随机划分由 run_id 决定）
SPLIT_RATIO = "0.3"    # 读取 split0.3 数据文件（train 70% / test 30%）

DATASETS = ["fish", "concrete", "energy", "parkinsons",
            "temperature", "air", "skillcraft", "mi", "wave"]
RUN_IDS = list(range(10))  # 0..9


# ---------------------------------------------------------------------------
# Toy 实验配置（train/_2d：数据在线生成，无需 preprocess，循环变量是 seed）
# ---------------------------------------------------------------------------
TOY_ENTRY = os.path.join(PROJECT_ROOT, "train", "_2d", "test_2d_parallel_org_space.py")
TOY_DATASETS = ["squares", "four_triangle", "elastic_ring", "half_gauss", "gauss_stick"]
TOY_SEEDS = list(range(10))       # 跑 10 个 seed
TOY_SPLIT_RATIO = "0.5"           # toy 里 train_size = int(data_num * split_ratio)


# ---------------------------------------------------------------------------
# 每个模型的超参数（原样拼进命令行）
# ---------------------------------------------------------------------------
def cdf_args():
    """cdf-first：CDFValNetMadeSMN（MADE 单调 CDF 网络）。"""
    return [
        "--total_epoch=20000",
        "--batch_size=256",
        "--lr=3e-4",
        "--grad_dx=0.01",
        "--early_stop_patience=1000",
        # "--dim_condition_mul=8,8,2",   # 每层条件维乘子（8×C, 8×C, 2×C）
        # "--dim_feature=16,16,1",       # 每层特征维（[1,16,16,1]，最后一维恒为 1）
        # "--num_group=32",
        # "--group_size=32",
        "--dim_feature=4,2,1",     
        "--dim_condition_mul=2,2,1",    
        "--num_group=4",
        "--group_size=4",
        "--use_bn=t",
        "--residual=f",

        "--min_max=soft",
        "--init_logsigma=-2",
        "--vl_loss_condition=0.01",
        "--vl_loss_point=0.005",
    ]




# net_name -> 参数构造函数（注意：这些函数不能带 point_dim 之类的数据集相关参数，
# 因为 cdf 的 dim_feature 最后一维对所有数据集恒为 1）
MODELS = {
    "cdf": cdf_args,
}
REAL_DEFAULT_MODELS = ["cdf"]


# ---------------------------------------------------------------------------
# Toy 模型的超参数（toy 数据是条件分布，必须 --condition=y，
# 否则走 draw_pic 非条件分支会因 cond/point 维度不匹配崩溃）
# ---------------------------------------------------------------------------
def toy_cdf_args():
    return [
        "--condition=y",
        "--total_epoch=20000",
        "--batch_size=256",
        "--lr=3e-4",
        "--grad_dx=0.01",
        # "--dim_condition_mul=8,8,2",
        # "--dim_feature=16,16,1",
        # "--num_group=32",
        # "--group_size=32",
        "--dim_feature=4,4,1",
        "--dim_condition_mul=4,2,1",    #parameters=12k
       
        "--num_group=4",
        "--group_size=4",
        "--use_bn=t",
        "--residual=f",
        "--min_max=soft",
        "--init_logsigma=-2",
        "--vl_loss_condition=0.05",
        "--vl_loss_point=0.01",
    ]




# toy 全部可选模型。cdf/hint/realnvp/nlcdf/atm/hrli 已实测；
# maf/nsf/raf/mdn/ddn 走同一套模型无关循环（get_net 已支持），但未在 toy 上实测，
# 其中 ddn 训练时不记录 train-nll（is_rec_train_nll 返回 False）。
TOY_MODELS = {
    "cdf": toy_cdf_args,
}


TOY_DEFAULT_MODELS = ["cdf"]

LOG_DIR = os.path.join(PROJECT_ROOT, "run_logs")
DONE_FILE = os.path.join(LOG_DIR, ".done.txt")
TOY_LOG_DIR = os.path.join(PROJECT_ROOT, "run_logs", "toy")
TOY_DONE_FILE = os.path.join(TOY_LOG_DIR, ".done.txt")


# ---------------------------------------------------------------------------
# 断点续跑：用一个 done 文件记录已完成任务
# ---------------------------------------------------------------------------
def run_key(net_name, data_name, run_id):
    return f"{net_name}\t{data_name}\t{run_id}"


def load_done(path=DONE_FILE):
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(line)
    return done


def mark_done(key, path=DONE_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(key + "\n")


# ---------------------------------------------------------------------------
# 命令构造 / 预处理
# ---------------------------------------------------------------------------
def build_command(net_name, data_name, run_id, device, seed, split_ratio):
    args = [
        sys.executable, ENTRY,
        f"--net_name={net_name}",
        f"--data_name={data_name}",
        f"--run_id={run_id}",
        f"--seed={seed}",
        f"--split_ratio={split_ratio}",
        f"--device={device}",
    ]
    args += MODELS[net_name]()
    return args


def build_toy_command(net_name, data_name, seed, device, data_num):
    args = [
        sys.executable, TOY_ENTRY,
        f"--net_name={net_name}",
        f"--data_name={data_name}",
        f"--seed={seed}",
        f"--split_ratio={TOY_SPLIT_RATIO}",
        f"--device={device}",
    ]
    args += TOY_MODELS[net_name]()
    args.append(f"--data_num={data_num}")
    return args


def preprocess(datasets, split_ratio):
    print(f"生成数据划分 (train_test_split_ratio={split_ratio}, K=10) ...")
    for d in datasets:
        cmd = [sys.executable, "-m", "data.from_ddn.pre_process",
               "--dataset", d, "--train_test_split_ratio", split_ratio]
        print("  $", " ".join(cmd))
        subprocess.run(cmd, cwd=PROJECT_ROOT)
    print("预处理完成\n")


# ---------------------------------------------------------------------------
# 结果汇总
# ---------------------------------------------------------------------------
def collect():
    """扫描 exp/real_data 下所有 log.log，提取 best val_nll 并打印表格。"""
    pattern = os.path.join(PROJECT_ROOT, "exp", "real_data", "*", "*", "*", "log.log")
    rows = []
    for path in glob.glob(pattern):
        rel = os.path.relpath(path, os.path.join(PROJECT_ROOT, "exp", "real_data"))
        parts = rel.split(os.sep)  # [data-split, net-argstr, seed-runid, log.log]
        if len(parts) != 4:
            continue
        data = parts[0].split("-")[0]
        net = parts[1].split("-")[0]
        run_id = parts[2].rsplit("-", 1)[-1]
        best = None
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = re.search(r"val_nll=([0-9.eE+\-]+)", line)
                if m:
                    v = float(m.group(1))
                    best = v if best is None else min(best, v)
        if best is not None:
            rows.append((net, data, int(run_id), best))

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    if not rows:
        print("没有找到任何结果（先运行训练，或检查 exp/real_data 目录）")
        return

    print(f"{'model':<9}{'dataset':<12}{'run':<5}val_nll")
    print("-" * 45)
    for net, data, rid, v in rows:
        print(f"{net:<9}{data:<12}{rid:<5}{v:.4f}")


# ---------------------------------------------------------------------------
# 参数量统计
# ---------------------------------------------------------------------------
def count_params(models, datasets, device="cpu", split_ratio=SPLIT_RATIO):
    """为每个 (模型, 数据集) 组合构建一次网络并统计参数量。"""
    print(f"{'model':<9}{'dataset':<12}{'cond':<7}{'point':<7}{'params':<12}{'trainable'}")
    print("-" * 62)
    for net_name in models:
        for data_name in datasets:
            cmd = build_command(net_name, data_name, 0, device, SEED, split_ratio)
            cmd.append("--count_params")
            ret = subprocess.run(cmd, cwd=PROJECT_ROOT,
                                 capture_output=True, text=True,
                                 encoding="utf-8", errors="replace")
            out = ret.stdout + ret.stderr
            m = re.search(r"\[PARAMS\] (.+)", out)
            if m:
                fields = dict(kv.split("=") for kv in m.group(1).split())
                cond = fields.get("condition_dim", "?")
                point = fields.get("point_dim", "?")
                total = fields.get("total", "?")
                trainable = fields.get("trainable", "?")
                print(f"{net_name:<9}{data_name:<12}{cond:<7}{point:<7}{total:<12}{trainable}")
            else:
                print(f"{net_name:<9}{data_name:<12}{'-':<7}{'-':<7}  <构建失败 rc={ret.returncode}>")
                tail = "\n".join(out.strip().splitlines()[-15:])
                if tail:
                    print("    " + tail.replace("\n", "\n    "))


def count_toy_params(models, datasets, device="cpu"):
    """Build toy models and print parameter counts without training."""
    print(f"{'model':<9}{'dataset':<14}{'cond':<7}{'point':<7}{'params':<12}{'trainable'}")
    print("-" * 68)
    for net_name in models:
        for data_name in datasets:
            cmd = build_toy_command(net_name, data_name, 0, device, 1000)
            cmd.append("--count_params")
            ret = subprocess.run(cmd, cwd=PROJECT_ROOT,
                                 capture_output=True, text=True,
                                 encoding="utf-8", errors="replace")
            out = ret.stdout + ret.stderr
            m = re.search(r"\[PARAMS\] (.+)", out)
            if m:
                fields = dict(kv.split("=") for kv in m.group(1).split())
                cond = fields.get("condition_dim", "?")
                point = fields.get("point_dim", "?")
                total = fields.get("total", "?")
                trainable = fields.get("trainable", "?")
                print(f"{net_name:<9}{data_name:<14}{cond:<7}{point:<7}{total:<12}{trainable}")
            else:
                print(f"{net_name:<9}{data_name:<14}{'-':<7}{'-':<7}  <构建失败 rc={ret.returncode}>")
                tail = "\n".join(out.strip().splitlines()[-15:])
                if tail:
                    print("    " + tail.replace("\n", "\n    "))


# ---------------------------------------------------------------------------
# Toy 批量运行
# ---------------------------------------------------------------------------
def run_toy(models, datasets, seeds, data_nums, device, dry_run, no_skip):
    done = load_done(TOY_DONE_FILE)
    total = len(models) * len(datasets) * len(seeds) * len(data_nums)
    n_ok = n_fail = n_skip = 0

    print(f"Toy 任务总数 {total} = {len(models)} 模型 × {len(datasets)} 数据集 × {len(seeds)} seed × {len(data_nums)} data_num")
    print(f"device={device}  split={TOY_SPLIT_RATIO}  data_num={','.join(map(str, data_nums))}")
    print("-" * 70)

    for net_name in models:
        for data_name in datasets:
            for seed in seeds:
                for data_num in data_nums:
                    key = f"{net_name}\t{data_name}\t{seed}\t{data_num}"
                    tag = f"{net_name:<8} {data_name:<14} seed{seed} N={data_num}"
                    if key in done and not no_skip:
                        n_skip += 1
                        print(f"[skip] {tag}")
                        continue

                    cmd = build_toy_command(net_name, data_name, seed, device, data_num)
                    if dry_run:
                        print(f"[run ] {tag}")
                        print("      ", " ".join(cmd))
                        continue

                    print(f"[run ] {tag}", flush=True)
                    os.makedirs(TOY_LOG_DIR, exist_ok=True)
                    log_path = os.path.join(TOY_LOG_DIR, f"{net_name}__{data_name}__seed{seed}__N{data_num}.log")
                    t0 = time.time()
                    with open(log_path, "w", encoding="utf-8") as logf:
                        ret = subprocess.run(cmd, cwd=PROJECT_ROOT,
                                             stdout=logf, stderr=subprocess.STDOUT)
                    dt = time.time() - t0
                    if ret.returncode == 0:
                        mark_done(key, TOY_DONE_FILE)
                        n_ok += 1
                        print(f"       ok   ({dt:.0f}s)")
                    else:
                        n_fail += 1
                        print(f"       FAIL rc={ret.returncode} ({dt:.0f}s) -> {log_path}")

    print("-" * 70)
    print(f"成功 {n_ok} / 失败 {n_fail} / 跳过 {n_skip}")
    if dry_run:
        print("(dry-run，未真正执行)")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="批量运行条件密度模型（real/toy")
    ap.add_argument("--models", default=None, help="")
    ap.add_argument("--datasets", default=None, help="逗号分隔（默认 real 9 个 / toy 5 个）")
    ap.add_argument("--run_ids", default=",".join(map(str, RUN_IDS)), help="real 模式：逗号分隔 run_id")
    ap.add_argument("--seeds", default=",".join(map(str, TOY_SEEDS)), help="toy 模式：逗号分隔 seed")
    ap.add_argument("--data_nums", default="1000", help="toy 模式：逗号分隔 data_num（样本数）")
    ap.add_argument("--toy", action="store_true", help="跑 toy 实验（train/_2d，循环变量是 seed）")
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--split_ratio", default=SPLIT_RATIO, help="real 模式测试集比例（train 1-ratio / test ratio）")
    ap.add_argument("--preprocess", action="store_true", help="先跑 pre_process")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    ap.add_argument("--no-skip", action="store_true", help="已完成也重跑")
    ap.add_argument("--collect", action="store_true", help="只汇总已有结果（real 模式）")
    ap.add_argument("--count_params", action="store_true", help="只统计各模型参数量（不训练）")
    opt = ap.parse_args()

    if opt.toy:
        model_list = opt.models or ",".join(TOY_DEFAULT_MODELS)
        dataset_list = opt.datasets or ",".join(TOY_DATASETS)
        models = [m.strip() for m in model_list.split(",") if m.strip()]
        datasets = [d.strip() for d in dataset_list.split(",") if d.strip()]
        seeds = [int(s.strip()) for s in opt.seeds.split(",") if s.strip()]
        data_nums = [int(n.strip()) for n in opt.data_nums.split(",") if n.strip()]
        for m in models:
            assert m in TOY_MODELS, f"未知 toy 模型 {m}，可选 {list(TOY_MODELS)}"
        for d in datasets:
            assert d in TOY_DATASETS, f"未知 toy 数据集 {d}，可选 {TOY_DATASETS}"
        if opt.count_params:
            count_toy_params(models, datasets, "cpu")
            return
        run_toy(models, datasets, seeds, data_nums, opt.device, opt.dry_run, opt.no_skip)
        return

    if opt.collect:
        collect()
        return

    model_list = opt.models or ",".join(REAL_DEFAULT_MODELS)
    dataset_list = opt.datasets or ",".join(DATASETS)
    models = [m.strip() for m in model_list.split(",") if m.strip()]
    datasets = [d.strip() for d in dataset_list.split(",") if d.strip()]
    run_ids = [int(r.strip()) for r in opt.run_ids.split(",") if r.strip()]

    for m in models:
        assert m in MODELS, f"未知模型 {m}，可选 {list(MODELS)}"
    for d in datasets:
        assert d in DATASETS, f"未知数据集 {d}，可选 {DATASETS}"

    if opt.count_params:
        count_params(models, datasets, "cpu", opt.split_ratio)
        return

    if opt.preprocess and not opt.dry_run:
        preprocess(datasets, opt.split_ratio)

    done = load_done()
    total = len(models) * len(datasets) * len(run_ids)
    n_ok = n_fail = n_skip = 0

    print(f"任务总数 {total} = {len(models)} 模型 × {len(datasets)} 数据集 × {len(run_ids)} run_id")
    print(f"device={opt.device}  seed={opt.seed}  split={opt.split_ratio}")
    print("-" * 70)

    for net_name in models:
        for data_name in datasets:
            for run_id in run_ids:
                key = run_key(net_name, data_name, run_id)
                tag = f"{net_name:<8} {data_name:<11} run{run_id}"
                if key in done and not opt.no_skip:
                    n_skip += 1
                    print(f"[skip] {tag}")
                    continue

                cmd = build_command(net_name, data_name, run_id, opt.device, opt.seed, opt.split_ratio)
                if opt.dry_run:
                    print(f"[run ] {tag}")
                    print("      ", " ".join(cmd))
                    continue

                print(f"[run ] {tag}", flush=True)
                os.makedirs(LOG_DIR, exist_ok=True)
                log_path = os.path.join(LOG_DIR, f"{net_name}__{data_name}__run{run_id}.log")
                t0 = time.time()
                with open(log_path, "w", encoding="utf-8") as logf:
                    ret = subprocess.run(cmd, cwd=PROJECT_ROOT,
                                         stdout=logf, stderr=subprocess.STDOUT)
                dt = time.time() - t0
                if ret.returncode == 0:
                    mark_done(key)
                    n_ok += 1
                    print(f"       ok   ({dt:.0f}s)")
                else:
                    n_fail += 1
                    print(f"       FAIL rc={ret.returncode} ({dt:.0f}s) -> {log_path}")

    print("-" * 70)
    print(f"成功 {n_ok} / 失败 {n_fail} / 跳过 {n_skip}")
    if opt.dry_run:
        print("(dry-run，未真正执行)")


if __name__ == "__main__":
    main()
