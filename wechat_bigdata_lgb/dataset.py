import numpy as np

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

file_path = "data/offline_train/offline_train_read_comment_12_concate_sample.csv"

target = 'read_comment'
df = pd.read_csv(file_path)
train_target = torch.tensor(df[target].values.astype(np.float32))
train = torch.tensor(df.drop(target, axis=1).values.astype(np.float32))
train_dataset = TensorDataset(train, train_target)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
for i, (x, y) in enumerate(train_loader):
    print(i)
    print(x)
    print(y)
    assert False
