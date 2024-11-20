from torch.utils.data import Dataset
import polars as pl
import os
import sys
import torch

import data.read_data_polars as rdp
import data.cwt_transform as cwt


class DatasetPolars(Dataset):
    def __init__(self, df: pl.DataFrame, sequence_length: int = 640):
        self.df = df
        self.sequence_length = sequence_length
        self.cwt_transform = cwt.CwtTransform(
            fmin=1, fmax=200, n_frex=20, wavelet_type="cgau4", seq_length=sequence_length
        )

    def __len__(self):
        return self.df.height - self.sequence_length

    def get_target(self, idx):
        return torch.tensor(self.df[idx]["target"].to_numpy(), dtype=torch.long)

    def getdata(self, idx):
        return torch.tensor(self.df[idx: idx + self.sequence_length, :-1].to_numpy(),
                            dtype=torch.float32)

    def __getitem__(self, idx):
        return self.getdata(idx), self.get_target(idx)


