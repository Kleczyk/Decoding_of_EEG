from torch.utils.data import Dataset


class CWTSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        row = self.dataset.__getitem__(int(self.indices[idx]))
        return row

    def __len__(self):
        return len(self.indices)
