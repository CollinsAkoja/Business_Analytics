import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

NUMERIC = ["quantity","price"]
CATEGORICAL = ["store_id","payment_type","promo_flag"]

class AnomalyDataset(Dataset):
    def __init__(self, csv_path, scaler=None):
        data = pd.read_csv(csv_path, parse_dates=["timestamp"])
        data = self._preprocess(df)
        data["amount"] = data["price"] * data["quantity"]
        z = (data["amount"] - data["amount"].mean()) / (data["amount"].std() + 1e-9)
        data["label"] = (z.abs() > 3).astype(int)

        num = data[NUMERIC].fillna(data[NUMERIC].median())
        if scaler is None:
            self.scaler = StandardScaler().fit(num)
        else:
            self.scaler = scalerpython
        num_s = self.scaler.transform(num)
        store_ohe = pd.get_dummies(df["store_id"], prefix="store")
        pay_ohe = pd.get_dummies(df["payment_type"], prefix="pay")
        promo = data["promo_flag"].values.reshape(-1,1)
        X = np.hstack([num_s, store_ohe.values, pay_ohe.values, promo])
        self.X = X.astype(np.float32)
        self.y = data["label"].values.astype(np.int64)

    def _preprocess(self, data):
        data['hour'] = data["timestamp"].dt.hour
        data['dow'] = data["timestamp"].dt.dayofweek
        return data

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])
