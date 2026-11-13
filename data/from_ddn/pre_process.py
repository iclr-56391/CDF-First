import argparse
import torch
import os

from utils import PathAppend

def split_train_test(train_size, test_size):
    perm = torch.randperm(train_size + test_size)
    train_idx = perm[:train_size]
    test_idx = perm[train_size:]
    return train_idx, test_idx

def main(specify_dataset:None|str =None, split_ratio:float|None =None):
    PathAppend(f"{__file__}/../")
    from from_ddn.uci import UCIDataset

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['fish', 'concrete', 'energy', 'music', 'parkinsons', 'temperature', 'air', 'drug', 'skillcraft', 'mi', 'wave'], default="DATASET_NOT_CHOSEN")
    parser.add_argument('--train_test_split_ratio', type=float, default=0.3, help='测试集比例：0.3 → train 0.7 / test 0.3')
    parser.add_argument('--feature_drop_rate', type=float, default=0.0, choices=[0.0, 0.3, 0.5, 0.7])
    parser.add_argument('--K', type=int, default=10)
    
    args = parser.parse_known_args()[0]
    if specify_dataset is not None:
        args.dataset = specify_dataset
    args.train_test_split_ratio = split_ratio if split_ratio is not None else args.train_test_split_ratio

    torch.manual_seed(1234)

    dataset = UCIDataset(args.dataset, f"{__file__}/../../")    
    print(dataset.x_dim, dataset.y_dim)    

    for i in range(args.K):
        assert dataset is not None
        assert dataset.data is not None
        assert dataset.x_dim is not None
        assert dataset.targets is not None

        selected_feature = torch.randperm(dataset.x_dim)[:int(dataset.x_dim * (1 - args.feature_drop_rate))]
        data = dataset.data[:, selected_feature]
        data = (data - torch.mean(data, dim=0)) / torch.std(data, dim=0)    
        targets = dataset.targets
        targets = (targets - torch.mean(targets, dim=0)) / torch.std(targets, dim=0)

        # split_ratio 表示「测试集比例」：0.3 → train 70% / test 30%
        test_size = int(dataset.data.shape[0] * args.train_test_split_ratio)
        train_size = dataset.data.shape[0] - test_size

        train_idx, test_idx = split_train_test(train_size, test_size)
        train_X, train_Y, test_X, test_Y = data[train_idx], targets[train_idx],  data[test_idx], targets[test_idx]

        save_path = os.path.abspath(f'{__file__}/../../../__data/%s/split%g_drop%g_run%d.pt' % (args.dataset, args.train_test_split_ratio, args.feature_drop_rate, i))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save((train_X, train_Y, test_X, test_Y), save_path)


if __name__ == '__main__':   
    main()