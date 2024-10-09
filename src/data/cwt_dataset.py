from torch.utils.data import Dataset
from .db_contlorer import DbController


class CwtDataset(Dataset):
    def __init__(
        self, table: str, db_controller: DbController, sequence_length: int = 640
    ):
        self.table = table
        self.db_controller = db_controller
        self.sequence_length = sequence_length

    def __len__(self):
        return self.db_controller.get_len(self.table) - self.sequence_length

    def __getitem__(self, idx):
        return self.db_controller.get_data_between(
            self.table, idx, idx + self.sequence_length - 1
        )
