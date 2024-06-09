import numpy as np
import torch
import pickle
import psycopg2

class CWTDataset(torch.utils.data.Dataset):
    """
    Dataset for EEG data after CWT transformation stored in PostgreSQL database.
    """

    def __init__(self, engine, sequence_length=4000):
        """
        Constructor for CWTDataset class that initializes the dataset.
        Args:
            db_params (dict): Parameters to connect to the database.
            sequence_length (int): Length of the sequence.
        """
        self.engine = engine
        self.sequence_length = sequence_length
        self.conn = self.engine.connect()
        self.cursor = self.connengine.cursor()

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        self.conn.execute("SELECT COUNT(*) FROM wavelet_transforms")
        return self.conn.fetchone()[0] - self.sequence_length + 1

    def __getitem__(self, idx):
        """
        Function that returns as many samples as the sequence length.
        Args:
            idx (int): Index of the sample.
        Returns:
            tuple: (cwt_tensor, target_tensor)
        """
        query = "SELECT cwt_data, target FROM wavelet_transforms WHERE id >= %s AND id <= %s"
        self.conn.execute(query, (idx + 1, idx + self.sequence_length))
        rows = self.conn.fetchall()

        # Deserialize data and stack it into a numpy array
        cwt_sequence = np.stack([pickle.loads(row[0]) for row in rows])

        # The target from the last row of the fetched data
        target = rows[-1][1]

        # Convert numpy arrays to PyTorch tensors
        cwt_tensor = torch.tensor(cwt_sequence, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.int64)

        return cwt_tensor, target_tensor

    def __del__(self):
        """
        Destructor to close database connection and cursor.
        """
        self.cursor.close()
        self.conn.close()
