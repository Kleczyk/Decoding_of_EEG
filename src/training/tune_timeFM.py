import torch
from torch.utils.data import DataLoader, Dataset, random_split
import lightning.pytorch as pl
from transformers import AutoTokenizer
from models.base_timefm import TimesFMBaseLighting
# Dummy dataset for testing
class TimeSeriesDataset(Dataset):
    def __init__(self, num_samples: int, sequence_length: int, num_classes: int):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randint(0, 100, (self.sequence_length,))
        y = torch.randint(0, self.num_classes, (1,))
        return x, y.squeeze()

# Initialize model
def main():
    pretrained_model_name = "google/timesfm-1.0-200m"
    sequence_length = 128
    num_classes = 3
    learning_rate = 1e-4
    batch_size = 16
    num_epochs = 5

    # Prepare dataset and dataloaders
    dataset = TimeSeriesDataset(num_samples=1000, sequence_length=sequence_length, num_classes=num_classes)
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    model = TimesFMBaseLighting(
        pretrained_model_name=pretrained_model_name,
        sequence_length=sequence_length,
        num_classes=num_classes,
        learning_rate=learning_rate,
    )

    # Trainer setup
    trainer = pl.Trainer(max_epochs=num_epochs, accelerator="auto", log_every_n_steps=10)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()
