# src/corruption.py
import numpy as np
import cv2
from PIL import Image
import random
from skimage.util import random_noise
import os


class ImageCorruptor:
    """Generate various types of image corruptions for training"""

    def __init__(self):
        self.corruption_types = [
            "gaussian_noise",
            "motion_blur",
            "compression",
            "mixed",  # combination of corruptions
        ]

    def add_gaussian_noise(self, image, severity=0.1):
        """Add Gaussian noise to image"""
        if isinstance(image, np.ndarray):
            # Convert to 0-1 range for skimage
            img_float = image.astype(np.float32) / 255.0
            noisy = random_noise(img_float, mode="gaussian", var=severity**2)
            return (noisy * 255).astype(np.uint8)
        return image

    def add_motion_blur(self, image, severity=15):
        """Add motion blur using kernel"""
        if isinstance(image, np.ndarray):
            # Create motion blur kernel
            kernel_size = max(5, int(severity))
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size

            blurred = cv2.filter2D(image, -1, kernel)
            return blurred
        return image

    def add_compression_artifacts(self, image, quality=30):
        """Add JPEG compression artifacts"""
        if isinstance(image, np.ndarray):
            # Convert to PIL for JPEG compression
            pil_img = Image.fromarray(image)

            # Save to bytes with low quality
            import io

            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)

            # Load back with artifacts
            compressed_img = Image.open(buffer)
            return np.array(compressed_img)
        return image

    def corrupt_image(self, image, corruption_type, severity_level="medium"):
        """Apply specified corruption to image"""

        # Severity mappings
        severity_map = {
            "low": {"noise": 0.05, "blur": 10, "quality": 50},
            "medium": {"noise": 0.1, "blur": 15, "quality": 30},
            "high": {"noise": 0.2, "blur": 25, "quality": 15},
        }

        severities = severity_map[severity_level]

        if corruption_type == "gaussian_noise":
            return self.add_gaussian_noise(image, severities["noise"])

        elif corruption_type == "motion_blur":
            return self.add_motion_blur(image, severities["blur"])

        elif corruption_type == "compression":
            return self.add_compression_artifacts(image, severities["quality"])

        elif corruption_type == "mixed":
            # Apply 2-3 random corruptions
            corruptions = random.sample(
                ["gaussian_noise", "motion_blur", "compression"], 2
            )
            corrupted = image.copy()
            for corr in corruptions:
                corrupted = self.corrupt_image(corrupted, corr, severity_level)
            return corrupted

        else:
            return image  # Return original if unknown corruption type

    def generate_corrupted_dataset(self, image_dir, output_dir, csv_file):
        """Generate corrupted versions of all images in dataset"""

        import pandas as pd

        # Load image list
        df = pd.read_csv(csv_file)

        # Create output directories
        for corruption_type in ["clean"] + self.corruption_types:
            os.makedirs(f"{output_dir}/{corruption_type}", exist_ok=True)

        corrupted_data = []

        print("Generating corrupted images...")
        for idx, row in df.head(200).iterrows():  # Start with 200 images for speed
            img_path = os.path.join(image_dir, row["Image Index"])

            if not os.path.exists(img_path):
                continue

            # Load image
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            # Resize to standard size for consistency
            image = cv2.resize(image, (256, 256))

            # Save clean version
            clean_path = f"{output_dir}/clean/{row['Image Index']}"
            cv2.imwrite(clean_path, image)
            corrupted_data.append(
                {
                    "image_path": clean_path,
                    "corruption_type": "clean",
                    "severity": "none",
                    "original_file": row["Image Index"],
                }
            )

            # Generate corrupted versions
            for corruption_type in self.corruption_types:
                for severity in ["low", "medium", "high"]:
                    corrupted_img = self.corrupt_image(image, corruption_type, severity)

                    # Save corrupted image
                    filename = f"{corruption_type}_{severity}_{row['Image Index']}"
                    corrupted_path = f"{output_dir}/{corruption_type}/{filename}"
                    cv2.imwrite(corrupted_path, corrupted_img)

                    corrupted_data.append(
                        {
                            "image_path": corrupted_path,
                            "corruption_type": corruption_type,
                            "severity": severity,
                            "original_file": row["Image Index"],
                        }
                    )

            if idx % 20 == 0:
                print(f"Processed {idx + 1} images...")

        # Save metadata
        corrupted_df = pd.DataFrame(corrupted_data)
        corrupted_df.to_csv(f"{output_dir}/corrupted_dataset.csv", index=False)

        print(f"Generated {len(corrupted_data)} corrupted images")
        print(f"Corruption types: {corrupted_df['corruption_type'].value_counts()}")

        return corrupted_df


# Usage example
if __name__ == "__main__":
    corruptor = ImageCorruptor()

    # Generate corrupted dataset
    corrupted_df = corruptor.generate_corrupted_dataset(
        image_dir="../data/raw/images/",
        output_dir="../data/corrupted",
        csv_file="../data/processed/usable_subset.csv",
    )
