# Libraries
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from models import get_model


class CorruptionDetector:
    """Easy-to-use corruption detector"""

    def __init__(self, model_path, device=None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Class names
        self.class_names = [
            "clean",
            "gaussian_noise",
            "motion_blur",
            "compression",
            "mixed",
        ]

        # Load model
        self.model = get_model(num_classes=len(self.class_names), pretrained=False)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Transform for inference
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        print(f"✓ Model loaded from: {model_path}")
        print(f"✓ Running on: {self.device}")

    def predict_image(self, image_path):
        """Predict corruption type for a single image"""

        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Transform
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = self.class_names[predicted.item()]
        confidence_score = confidence.item() * 100

        # Get all class probabilities
        all_probs = {
            self.class_names[i]: probabilities[0][i].item() * 100
            for i in range(len(self.class_names))
        }

        return {
            "predicted_class": predicted_class,
            "confidence": confidence_score,
            "all_probabilities": all_probs,
            "image": image,
        }

    def visualize_prediction(self, image_path, save_path=None):
        """Visualize prediction with probabilities"""

        result = self.predict_image(image_path)

        # Create figure
        fig = plt.figure(figsize=(12, 5))

        # Left: Image
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(result["image"], cmap="gray")
        ax1.axis("off")
        ax1.set_title(
            f"Predicted: {result['predicted_class'].upper()}\n"
            f"Confidence: {result['confidence']:.1f}%",
            fontsize=12,
            fontweight="bold",
        )

        # Right: Probability bar chart
        ax2 = plt.subplot(1, 2, 2)
        classes = list(result["all_probabilities"].keys())
        probs = list(result["all_probabilities"].values())

        colors = [
            "green" if c == result["predicted_class"] else "lightblue" for c in classes
        ]

        bars = ax2.barh(classes, probs, color=colors)
        ax2.set_xlabel("Probability (%)", fontsize=11)
        ax2.set_title("Class Probabilities", fontsize=12, fontweight="bold")
        ax2.set_xlim(0, 100)

        # Add percentage labels
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax2.text(prob + 2, i, f"{prob:.1f}%", va="center", fontsize=9)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✓ Visualization saved to: {save_path}")

        plt.show()

        return result

    def batch_predict(self, image_dir, max_images=20):
        """Predict on multiple images"""

        image_files = [
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        results = []
        for img_file in image_files[:max_images]:
            img_path = os.path.join(image_dir, img_file)
            try:
                result = self.predict_image(img_path)
                result["filename"] = img_file
                results.append(result)
                print(
                    f"✓ {img_file}: {result['predicted_class']} ({result['confidence']:.1f}%)"
                )
            except Exception as e:
                print(f"✗ Failed to process {img_file}: {e}")

        return results

    def create_demo_grid(
        self, corrupted_data_csv, save_path="../results/demo_grid.png", n_samples=12
    ):
        """Create a grid showing model predictions on various corruptions"""

        import pandas as pd

        df = pd.read_csv(corrupted_data_csv)

        # Sample images from each class
        samples = []
        for class_name in self.class_names:
            class_samples = df[df["corruption_type"] == class_name].sample(
                min(3, len(df[df["corruption_type"] == class_name]))
            )
            samples.append(class_samples)

        samples_df = pd.concat(samples).sample(
            n=min(n_samples, len(pd.concat(samples)))
        )

        # Create grid
        n_cols = 4
        n_rows = (len(samples_df) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        for idx, (_, row) in enumerate(samples_df.iterrows()):
            if idx >= len(axes):
                break

            result = self.predict_image(row["image_path"])

            axes[idx].imshow(result["image"], cmap="gray")
            axes[idx].axis("off")

            true_label = row["corruption_type"]
            pred_label = result["predicted_class"]

            # Color code: green if correct, red if wrong
            color = "green" if true_label == pred_label else "red"

            title = f"True: {true_label}\nPred: {pred_label}\nConf: {result['confidence']:.1f}%"
            axes[idx].set_title(title, fontsize=9, color=color, fontweight="bold")

        # Hide unused subplots
        for idx in range(len(samples_df), len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            "Corruption Detection Demo - Model Predictions",
            fontsize=16,
            fontweight="bold",
            y=1.00,
        )
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Demo grid saved to: {save_path}")
        plt.show()


def main():
    """Demo the corruption detection model"""

    print("=" * 70)
    print("CORRUPTION DETECTION MODEL - INFERENCE DEMO")
    print("=" * 70)

    # Initialize detector
    model_path = "../models_saved/best_model.pth"

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please train the model first!")
        return

    detector = CorruptionDetector(model_path)

    # Demo 1: Create prediction grid
    print("\n1. Creating demo grid with predictions...")
    corrupted_csv = "../data/corrupted/corrupted_dataset.csv"
    if os.path.exists(corrupted_csv):
        detector.create_demo_grid(corrupted_csv)

    # Demo 2: Single image prediction with visualization
    print("\n2. Running single image demo...")

    # Find a sample image
    import pandas as pd

    df = pd.read_csv(corrupted_csv)

    # Test on different corruption types
    for corruption_type in [
        "clean",
        "gaussian_noise",
        "motion_blur",
        "compression",
        "mixed",
    ]:
        sample = df[df["corruption_type"] == corruption_type].iloc[0]
        print(f"\n--- Testing {corruption_type.upper()} image ---")
        result = detector.visualize_prediction(
            sample["image_path"],
            save_path=f"../results/prediction_{corruption_type}.png",
        )
        print(
            f"Result: {result['predicted_class']} ({result['confidence']:.1f}% confidence)"
        )

    print("\n" + "=" * 70)
    print("INFERENCE DEMO COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - ../results/demo_grid.png")
    print("  - ../results/prediction_*.png")


if __name__ == "__main__":
    main()
