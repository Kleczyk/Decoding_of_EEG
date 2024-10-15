from torch.utils.data import Dataset
import polars as pl
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.data.read_data_polars as rdp
import src.data.cwt_transform as cwt


class CwtDataset(Dataset):
    def __init__(self, df: pl.DataFrame, sequence_length: int = 640):
        self.df = df
        self.sequence_length = sequence_length
        self.cwt_transform = cwt.CwtTransform(
            fmin=1, fmax=200, n_frex=20, wavelet_type="cgau4", seq_length=sequence_length
        )

    def __len__(self):
        return self.df.height - self.sequence_length

    def __getitem__(self, idx):
        return torch.from_numpy(self.cwt_transform.transform(self.df[idx : idx + self.sequence_length]))
