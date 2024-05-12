from torch.utils.data import DataLoader
import torch
from src.data.CWTDataset import CWTDataset
from src.models.CWT_EEG import CWT_EEG

class CWT_EEG_CrossPersonValidation(CWT_EEG):
    def __init__(self, batch_size, sequence_length, input_size, hidden_size, num_layers, lr, label_smoothing=0):
        super().__init__(batch_size, sequence_length, input_size, hidden_size, num_layers, lr, label_smoothing)

    def setup(self, stage=None):
        self.train_set = CWTDataset("df_train_cwt_data.db", self.hparams.sequence_length)
        self.val_set = CWTDataset("df_val_cwt_data.db", self.hparams.sequence_length)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.hparams.batch_size, num_workers=14,
                                           shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.hparams.batch_size, num_workers=14)
