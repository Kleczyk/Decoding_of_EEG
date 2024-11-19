from torch.utils.data import Dataset
import polars as pl
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.data.read_data_polars as rdp
import src.data.cwt_transform as cwt


class CwtDatasetPolars(Dataset):
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
        return torch.tensor(self.cwt_transform.transform(self.df[idx: idx + self.sequence_length, : -1]),
                            dtype=torch.float32)

    def __getitem__(self, idx):
        return self.getdata(idx), self.get_target(idx)

# cwt = CwtDataset(rdp.read_all_file_df_polars(channels_names=["Fc5.", "Fc3.", "Fc1.", "Af3.","T7..", "T8..", "T9..", "T10.", "Tp7.", "Tp8.", "P7..", "P5..", "P3..", "P1..", "Pz..", "P2..", "P4..", "P6..", "P8..", "Po7.", "Po3.", "Poz.", "Po4.", "Po8.", "O1..", "Oz..", "O2..", "Iz.."], idx_exp=[1], idx_people=[1]), sequence_length=640)
# print(cwt[6][1].shape)
# print(cwt[6][0].shape)
