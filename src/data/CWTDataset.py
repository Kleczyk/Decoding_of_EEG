import numpy as np
from torch.utils.data import Dataset
import torch
import pickle
import sqlite3


class CWTDataset(Dataset):
    """
    Dataset for EEG data after CWT transformation stored in SQLite database.

    Attributes:
        db_path: str - path to SQLite database
        sequence_length: int - length of the sequence
    Methods:
        __len__ - returns the number of samples in the dataset minus the sequence length
        __getitem__ - returns a sample from the dataset

    """

    def __init__(self, db_path, sequence_length=4000):
        """
        Constructor for CWTDataset class that initializes the dataset.
        Args:
            db_path: str - path to SQLite database
            sequence_length: int - length of the sequence
        Returns:
            None
        """
        self.db_path = db_path
        self.sequence_length = sequence_length
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("SELECT COUNT(*) FROM wavelet_transforms")
        self.total_samples = self.cursor.fetchone()[0]

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        Args:
            None
        Returns:
            int - number of samples in the dataset minus the sequence length
        """
        return self.total_samples - self.sequence_length + 1

    def __getitem__(self, idx):
        """
        function that returns as many samples as the sequence length
        Args:
            idx: int - index of the sample
        Returns:
            tuple - (cwt_tensor, target_tensor)

        """
        query = (
            "SELECT cwt_data, target FROM wavelet_transforms WHERE id BETWEEN ? AND ?"
        )

        self.cursor.execute(query, (idx + 1, idx + self.sequence_length))
        rows = self.cursor.fetchall()

        cwt_sequence = np.stack([pickle.loads(row[0]) for row in rows])

        target = rows[-1][1]

        cwt_tensor = torch.tensor(cwt_sequence, dtype=torch.float32)

        target_tensor = torch.tensor(target, dtype=torch.int64)
        return cwt_tensor, target_tensor

    def __del__(self):
        self.conn.close()
