from torch.utils.data import DataLoader
import torch
from src.data.CWTDataset import CWTDataset
from src.models.CWT_EEG import CWT_EEG


class CWT_EEG_CrossPersonValidation(CWT_EEG):
    """
    Class for training the model using validation data from people who are not included in train set.
    Attributes:
        batch_size: int - number of samples in a batch
        sequence_length: int - length of the sequence
        input_size: int - number of features in the input
        hidden_size: int - number of features in the hidden state
        num_layers: int - number of LSTM layers
        lr: float - learning rate
        label_smoothing: float - label smoothing factor
    Methods:
        __init__ - constructor for the class
        setup - sets up the training and validation datasets
        train_dataloader - returns the training dataloader
        val_dataloader - returns the validation dataloader
    """

    def __init__(self, batch_size, sequence_length, input_size, hidden_size, num_layers, lr, label_smoothing=0,
                 conn_train=None, conn_val=None):
        super().__init__(batch_size, sequence_length, input_size, hidden_size, num_layers, lr, label_smoothing,
                         conn_train, conn_val)
        self.save_hyperparameters('batch_size', 'sequence_length', 'input_size', 'hidden_size', 'num_layers', 'lr',
                                  'label_smoothing')
        self.conn_train = conn_train
        self.conn_val = conn_val

    def setup(self, stage=None):
        self.train_set = CWTDataset(self.conn_train, self.hparams.sequence_length)
        self.val_set = CWTDataset(self.conn_val, self.hparams.sequence_length)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.hparams.batch_size, num_workers=14,
                                           shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.hparams.batch_size, num_workers=14)
