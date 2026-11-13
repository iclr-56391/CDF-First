
import sys
import os
import argparse
add_path = os.path.abspath(f"{__file__}/../../../")

print(f"[Add Path] {add_path}")
sys.path.append(add_path)

import data.from_ddn.pre_process

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=float, default=0.3, help='Train/test split ratio')
    args = parser.parse_args()
    
    datasets = ['fish', 'concrete', 'energy', 'music', 'parkinsons', 'temperature', 'air', 'drug', 'skillcraft']
    for dataset in datasets:
        data.from_ddn.pre_process.main(dataset, split_ratio=args.split)