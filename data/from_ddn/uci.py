import os
import torch
import urllib
import pandas as pd
import zipfile
import numpy as np

from torch.utils import data
import urllib.request

class UCIDataset():
    def __init__(self, name, data_path='data'):
        self.datasets = {
            'concrete': 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls',
            'fish': 'http://archive.ics.uci.edu/ml/machine-learning-databases/00504/qsar_fish_toxicity.csv',

            'energy': 'http://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx',
            'music': 'http://archive.ics.uci.edu/ml/machine-learning-databases/00315/Geographical%20Original%20of%20Music.zip',

            'parkinsons': 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data',
            'temperature': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00514/Bias_correction_ucl.csv',

            'air' : 'https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip',
            'drug': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data',
            'skillcraft': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00272/SkillCraft1_Dataset.csv',
            'mi': 'https://archive.ics.uci.edu/static/public/579/data.csv',
            'wave': 'https://archive.ics.uci.edu/static/public/882/large-scale+wave+energy+farm.zip'
        }
        self.data_path = os.path.dirname(os.path.abspath(data_path))
        print(f"DataDir is {data_path}")
        self.name = name
        self._load_dataset()
    
    def _load_dataset(self):
        data = None
        if self.name not in self.datasets:
            raise Exception('Unknown dataset!')

        self.x_dim = None
        self.data = None
        self.targets = None
        self.y_dim = None
        url = self.datasets[self.name]
        file = os.path.join(self.data_path, 'UCI', url.split('/')[-1])
        file = os.path.abspath(file)
        print(f"Processing... {file}")
        os.makedirs(os.path.dirname(os.path.abspath(file)), exist_ok=True)

        if not os.path.exists(file):
            print(f"Downloading... {self.name}")
            urllib.request.urlretrieve(url, file)
            print(f"Download finish {self.name}")

        if self.name == 'boston':
            data = pd.read_csv(file, header=None, delimiter='\s+').values
        elif self.name == 'concrete':
            data = pd.read_excel(file, header=0).values
        elif self.name == 'energy':
            data = pd.read_excel(file, header=0, engine='openpyxl').values[:768, :10]
            self.x_dim = data.shape[1] - 2
            self.y_dim = 2
        elif self.name == 'air':
            dim = 3
            zip_dir = os.path.join(self.data_path, 'UCI', 'air')
            if not os.path.exists(zip_dir):
                zipfile.ZipFile(file).extractall(os.path.join(self.data_path, 'UCI', 'air'))
            file = os.path.join(self.data_path, 'UCI', 'air', 'AirQualityUCI.xlsx')
            data = pd.read_excel(file, header=0, engine='openpyxl').values[:, 2:-2].astype(np.float32)
            data[:, -dim:][data[:, -dim:] == -200] = np.nan 
            data = data[~np.isnan(data).any(axis=1)]
            data = torch.from_numpy(data)
            self.data = data[:, :-dim]
            self.targets = data[:, -dim:]
            self.x_dim = data.shape[1] - dim
            self.y_dim = dim
        elif self.name == 'fish':
            data = pd.read_csv(file, header=None, delimiter=';').values
        elif self.name == 'music':
            zip_dir = os.path.join(self.data_path, 'UCI', 'music')
            if not os.path.exists(zip_dir):
                zipfile.ZipFile(file).extractall(os.path.join(self.data_path, 'UCI', 'music'))
            file = os.path.join(self.data_path, 'UCI', 'music', 'Geographical Original of Music', 'default_features_1059_tracks.txt')
            data = pd.read_csv(file, header=0).values
            self.x_dim = data.shape[1] - 2
            self.y_dim = 2
        elif self.name == 'parkinsons':
            data = pd.read_csv(file, header=0, delimiter=',').values[:, 4:]
            data = torch.from_numpy(data).float()
            self.data = data[:, 2:]
            self.targets = data[:, :2]
            self.x_dim = data.shape[1] - 2
            self.y_dim = 2
        elif self.name == 'drug':
            data = pd.read_csv(file, header=None, delimiter=',')
            targets = data.iloc[:, 1:-19].values.astype(float)
            data = data.iloc[:, -19:].replace(regex=['CL'], value='').values.astype(float)
            self.x_dim = data.shape[1]
            self.y_dim = targets.shape[1]

            self.targets = torch.from_numpy(targets).float()
            self.data = torch.from_numpy(data).float()

        elif self.name == 'temperature':
            data = pd.read_csv(file, header=0, delimiter=',')
            data = data.replace('NaN', np.nan).dropna().values[:, 2:].astype(np.float32)
            data = torch.from_numpy(data)
            self.data = data[:, :-2]
            self.targets = data[:, -2:]
            self.x_dim = data.shape[1] - 2
            self.y_dim = 2          

        elif self.name == 'skillcraft':
            data = pd.read_csv(file, header=0, delimiter=',').replace(
                '?', np.nan).dropna().values[:, 1:].astype(np.float32)
            data = torch.from_numpy(data)
            self.targets = data[:, 10:14]
            self.data = torch.cat([data[:, 0:10], data[:, 14:]], dim=1)
            self.x_dim = self.data.shape[1]
            self.y_dim = self.targets.shape[1]

        elif self.name == 'mi':
            # Myocardial infarction complications (UCI id 579).
            # 124 cols = ID + 111 conditions + 12 targets (all 12 targets kept).
            # Conditions reduced 111 -> 48: drop features with >30% missing
            # values, or whose dominant value covers >=95% of non-missing data
            # (near-constant, no signal). Remaining missing values imputed:
            # median for continuous (>10 unique), mode for binary/categorical.
            df = pd.read_csv(file, header=0)
            data = df.values  # satisfy the trailing `assert data is not None`
            target_cols = ['FIBR_PREDS', 'PREDS_TAH', 'JELUD_TAH', 'FIBR_JELUD',
                           'A_V_BLOK', 'OTEK_LANC', 'RAZRIV', 'DRESSLER', 'ZSN',
                           'REC_IM', 'P_IM_STEN', 'LET_IS']
            feature_cols = [c for c in df.columns if c != 'ID' and c not in target_cols]
            X = df[feature_cols]
            miss_rate = X.isna().mean()
            dom = X.apply(lambda s: (s.value_counts(dropna=True).iloc[0] / s.count()) if s.count() else 1.0)
            keep = [c for c in feature_cols if miss_rate[c] <= 0.30 and dom[c] < 0.95]
            X = X[keep].copy()
            for c in keep:
                if X[c].nunique(dropna=True) > 10:
                    X[c] = X[c].fillna(X[c].median())
                else:
                    X[c] = X[c].fillna(X[c].mode()[0])
            self.data = torch.from_numpy(X.to_numpy(dtype=np.float32))
            self.targets = torch.from_numpy(df[target_cols].to_numpy(dtype=np.float32))
            self.x_dim = self.data.shape[1]
            self.y_dim = self.targets.shape[1]

        elif self.name == 'wave':
            # Large-scale Wave Energy Farm (UCI id 882). Uses WEC_Perth_49.csv
            # (49 WECs, Perth scenario; ~36k rows, 149 cols, no missing values).
            # condition (x) = first 10 WEC coordinates: X1,Y1..X10,Y10 (20 dims)
            # target   (y) = Total_Power + Power1..Power9 (10 dims)
            import zipfile
            extract_dir = os.path.join(os.path.dirname(file), 'wave')
            csv_path = os.path.join(extract_dir, 'WEC_Perth_49.csv')
            if not os.path.exists(csv_path):
                os.makedirs(extract_dir, exist_ok=True)
                zipfile.ZipFile(file).extract('WEC_Perth_49.csv', extract_dir)
            df = pd.read_csv(csv_path)
            # 只用 2000 条（固定随机种子可复现；整份文件 ~36k 行）
            if len(df) > 2000:
                df = df.sample(n=2000, random_state=0).reset_index(drop=True)
            data = df.values  # satisfy the trailing `assert data is not None`
            target_cols = ['Total_Power'] + [f'Power{i}' for i in range(1, 10)]
            condition_cols = [f'{a}{i}' for i in range(1, 11) for a in ('X', 'Y')]
            self.data = torch.from_numpy(df[condition_cols].to_numpy(dtype=np.float32))
            self.targets = torch.from_numpy(df[target_cols].to_numpy(dtype=np.float32))
            self.x_dim = self.data.shape[1]
            self.y_dim = self.targets.shape[1]

        assert data is not None
        if self.x_dim == None:
            self.x_dim = data.shape[1] - 1
        if self.y_dim == None:
            self.y_dim = 1        
        if self.data == None:
            self.data = torch.from_numpy(data[:, :self.x_dim]).float()
        if self.targets == None:
            self.targets = torch.from_numpy(data[:, -self.y_dim:]).float()


if __name__ == '__main__':   
    d = UCIDataset('skillcraft')
    assert d.targets is not None
    print(d.x_dim, d.y_dim, d.targets.shape[0])