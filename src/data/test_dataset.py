from torch.utils.data import Dataset, DataLoader
import torch

GLOBAL_CHANNEL_NAMES = [
    "Fc5.", "Fc3.", "Fc1.", "Af3.", "Afz.", "Af4.", "Af8.",
    "F7..", "F5..", "F3..", "F1..", "Fz..", "F2..", "F4..", "F6..", "F8..",
    "Ft7.", "Ft8.", "T7..", "T8..", "T9..", "T10.", "Tp7.", "Tp8.",
    "P7..", "P5..", "P3..", "P1..", "Pz..", "P2..", "P4..", "P6..", "P8..",
    "Po7.", "Po3.", "Poz.", "Po4.", "Po8.", "O1..", "Oz..", "O2..", "Iz..",
]
class TestEEGDataset(Dataset):
    """
    Custom Dataset for generating synthetic EEG data for testing.

    Args:
        num_samples (int): Number of samples in the dataset.
        seq_length (int): Sequence length for each sample.
        num_channels (int): Number of EEG channels.
        num_classes (int): Number of target classes.
    """
    def __init__(self, num_samples: int, seq_length: int, num_channels: int, num_classes: int):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.data = torch.rand(num_samples, seq_length, num_channels)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Example Usage
def get_test_dataloaders_via_dataset(config: dict) -> tuple[DataLoader, DataLoader]:
    """
    Generates train and validation DataLoaders using the TestEEGDataset.

    Args:
        config (dict): Configuration dictionary containing dataset parameters.

    Returns:
        tuple[DataLoader, DataLoader]: Train and validation DataLoaders.
    """
    num_samples_train = 10  # Small dataset for quick training
    num_samples_val = 5     # Small validation set
    num_channels = len(GLOBAL_CHANNEL_NAMES)

    train_dataset = TestEEGDataset(
        num_samples=num_samples_train,
        seq_length=config["seq_length"],
        num_channels=num_channels,
        num_classes=config["num_classes"]
    )
    val_dataset = TestEEGDataset(
        num_samples=num_samples_val,
        seq_length=config["seq_length"],
        num_channels=num_channels,
        num_classes=config["num_classes"]
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    return train_loader, val_loader

# Testing the Dataset
if __name__ == "__main__":
    # Example minimal configuration
    test_config = {
        "seq_length": 10,
        "batch_size": 2,
        "hidden_size": 16,
        "num_layers": 1,
        "dropout": 0.1,
        "lr": 0.01,
        "num_classes": 3,
        "max_epochs": 1,
    }

    train_loader, val_loader = get_test_dataloaders_via_dataset(test_config)
    for batch in train_loader:
        data, labels = batch
        print("Data shape:", data.shape)
        print("Labels shape:", labels.shape)
        break
