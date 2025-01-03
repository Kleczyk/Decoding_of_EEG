from torch.utils.data import Dataset
import pandas as pd
import torch
import data.cwt_transform as cwt


class BaseEEGDataset(Dataset):
    def __init__(self, df: pd.DataFrame, sequence_length: int = 640):
        self.df = df
        self.sequence_length = sequence_length
        self.cwt_transform = cwt.CwtTransform(
            fmin=1, fmax=200, n_frex=20, wavelet_type="cgau4", seq_length=sequence_length
        )
    def __len__(self):
        return self.df.shape[0] - self.sequence_length

    def get_target(self, idx):
        return torch.tensor(self.df.iloc[idx]["target"], dtype=torch.long)

    def getdata(self, idx):
        return torch.tensor(self.df.iloc[idx: idx + self.sequence_length, :-1].values,
                            dtype=torch.float32)

    def __getitem__(self, idx):
        return self.getdata(idx), self.get_target(idx)


