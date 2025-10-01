# Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import get_model
from src.data_utils import create_data_loaders


class DetectionTrainer:
    """Trainer class for corruption detection model"""

    def __init__(self, num_classes=5, learning_rate=1e-3, device=None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.num_classes = num_classes

        # Model
        self.model = get_model(num_classes=num_classes, pretrained=True).to(self.device)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_acc = 0.0

        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix(
                {"Loss": f"{loss.item():.4f}", "Acc": f"{100.0 * correct / total:.2f}%"}
            )

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for images, labels, _ in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Store for confusion matrix
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Update progress bar
                pbar.set_postfix(
                    {
                        "Loss": f"{loss.item():.4f}",
                        "Acc": f"{100.0 * correct / total:.2f}%",
                    }
                )

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc, all_predictions, all_labels

    def train(
        self, train_loader, val_loader, num_epochs=20, save_dir="../models_saved"
    ):
        """Full training loop"""

        os.makedirs(save_dir, exist_ok=True)

        print(f"\n{'=' * 50}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'=' * 50}\n")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # Learning rate schedule
            self.scheduler.step()

            # Print epoch summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_acc": val_acc,
                        "val_loss": val_loss,
                    },
                    f"{save_dir}/best_model.pth",
                )
                print(f"New model saved! (Val Acc: {val_acc:.2f}%)")

        print(f"\n{'=' * 50}")
        print("Training Complete!")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print(f"{'=' * 50}\n")

        return val_preds, val_labels

    def plot_training_curves(self, save_path="../results/training_curves.png"):
        """Plot training and validation curves"""

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss plot
        ax1.plot(self.train_losses, label="Train Loss", marker="o")
        ax1.plot(self.val_losses, label="Val Loss", marker="s")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True)

        # Accuracy plot
        ax2.plot(self.train_accuracies, label="Train Acc", marker="o")
        ax2.plot(self.val_accuracies, label="Val Acc", marker="s")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Training and Validation Accuracy")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved to: {save_path}")
        plt.show()

    def plot_confusion_matrix(
        self,
        predictions,
        labels,
        class_names,
        save_path="../results/confusion_matrix.png",
    ):
        """Plot confusion matrix"""

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        cm = confusion_matrix(labels, predictions)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")
        plt.show()

    def print_classification_report(self, predictions, labels, class_names):
        """Print detailed classification report"""

        report = classification_report(
            labels, predictions, target_names=class_names, digits=3
        )
        print("\n" + "=" * 50)
        print("Classification Report")
        print("=" * 50)
        print(report)


def main():
    """Main training function"""

    print("=" * 70)
    print("BIOMEDICAL IMAGE CORRUPTION DETECTION - MODEL TRAINING")
    print("=" * 70)

    # Configuration
    CONFIG = {
        "batch_size": 32,
        "num_epochs": 15,  # Start with 15 epochs for speed
        "learning_rate": 1e-3,
        "csv_file": "../data/corrupted/corrupted_dataset.csv",
    }

    print("\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")

    # Create data loaders
    print("\n" + "=" * 70)
    print("Loading Data...")
    print("=" * 70)

    train_loader, val_loader, class_names = create_data_loaders(
        csv_file=CONFIG["csv_file"], batch_size=CONFIG["batch_size"]
    )

    # Create trainer
    trainer = DetectionTrainer(
        num_classes=len(class_names), learning_rate=CONFIG["learning_rate"]
    )

    # Train model
    predictions, labels = trainer.train(
        train_loader, val_loader, num_epochs=CONFIG["num_epochs"]
    )

    # Plot results
    print("\n" + "=" * 70)
    print("Generating Visualizations...")
    print("=" * 70)

    trainer.plot_training_curves()
    trainer.plot_confusion_matrix(predictions, labels, class_names)
    trainer.print_classification_report(predictions, labels, class_names)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("Best model saved to: ../models_saved/best_model.pth")
    print("Results saved to: ../results/")


if __name__ == "__main__":
    main()
