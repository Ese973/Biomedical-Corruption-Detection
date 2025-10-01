# Libraries
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split
import os


class CorruptionDataset(Dataset):
    """Dataset class for corruption detection training"""

    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # Create label mapping
        self.corruption_types = [
            "clean",
            "gaussian_noise",
            "motion_blur",
            "compression",
            "mixed",
        ]
        self.label_to_idx = {
            label: idx for idx, label in enumerate(self.corruption_types)
        }
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load image
        image = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)

        if image is None:
            # Return zeros if image can't be loaded
            image = np.zeros((256, 256), dtype=np.uint8)

        # Convert to RGB (3 channels) for pretrained models
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Basic transform to tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Get label
        corruption_type = row["corruption_type"]
        label = self.label_to_idx[corruption_type]

        return image, label, row["image_path"]


def get_transforms(train=True):
    """Get image transforms for training/validation"""

    if train:
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),  # ResNet input size
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # ImageNet stats
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


def create_data_loaders(csv_file, batch_size=32, test_size=0.2, random_state=42):
    """Create train/validation data loaders"""

    # Load data and split
    df = pd.read_csv(csv_file)

    # Stratified split to ensure balanced classes
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["corruption_type"],
        random_state=random_state,
    )

    # Save splits for reference
    os.makedirs("../data/processed", exist_ok=True)
    train_df.to_csv("../data/processed/train_split.csv", index=False)
    val_df.to_csv("../data/processed/val_split.csv", index=False)

    # Create datasets
    train_dataset = CorruptionDataset(
        "../data/processed/train_split.csv", transform=get_transforms(train=True)
    )
    val_dataset = CorruptionDataset(
        "../data/processed/val_split.csv", transform=get_transforms(train=False)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset.corruption_types)}")
    print(f"Classes: {train_dataset.corruption_types}")

    return train_loader, val_loader, train_dataset.corruption_types


def visualize_batch(data_loader, num_samples=8):
    """Visualize a batch of images with their labels"""
    import matplotlib.pyplot as plt

    # Get a batch
    images, labels, paths = next(iter(data_loader))

    # Denormalize for visualization
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()

    for i in range(min(num_samples, len(images))):
        img = images[i]

        # Denormalize
        img = img * std[:, None, None] + mean[:, None, None]
        img = torch.clamp(img, 0, 1)

        # Convert to numpy and transpose
        img_np = img.numpy().transpose(1, 2, 0)

        axes[i].imshow(img_np)
        axes[i].set_title(f"Label: {labels[i].item()}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("../results/batch_visualization.png")
    plt.show()


# Test the data pipeline
if __name__ == "__main__":
    # Test data loading
    if os.path.exists("../data/corrupted/corrupted_dataset.csv"):
        train_loader, val_loader, classes = create_data_loaders(
            "../data/corrupted/corrupted_dataset.csv"
        )

        print("\nTesting data loader...")
        for batch_idx, (images, labels, paths) in enumerate(train_loader):
            print(f"Batch {batch_idx}: {images.shape}, {labels.shape}")
            if batch_idx == 0:  # Just test first batch
                visualize_batch(train_loader)
                break
    else:
        print("Run corruption generation first!")
