# Libraries
import os
import pandas as pd

# Assign data directory and image directory
csv_dir = "../data/raw/Data_Entry_2017.csv"
img_dir = "../data/raw/images/"


# Create function for usable subset of images
def create_usable_subset(df, img_dir, target_size=1000):
    """Create a subset of images from downloaded dataset"""

    available_images = []

    for image, row in df.iterrows():
        img_path = os.path.join(img_dir, row["Image Index"])
        if os.path.exists(img_path):
            available_images.append(image)

        # Stop if we have enough images
        if len(available_images) > target_size:
            print(f"Reached target of {target_size} images, stopping search.")
            break

    print(f"Total available images found: {len(available_images)}")

    if available_images:
        # Create subset dataframe
        df_subset = df.loc[available_images].copy()

        # Prioritize normal/clean images for corruption
        if "Finding Labels" in df_subset.columns:
            normal_images = df_subset[df_subset["Finding Labels"] == "No Finding"]
            other_images = df_subset[df_subset["Finding Labels"] != "No Finding"]

            print(f"Normal images (No Finding): {len(normal_images)}")
            print(f"Other findings: {len(other_images)}")

            # Use mostly normal images, some with findings for variety
            if len(normal_images) >= 800:
                final_subset = pd.concat(
                    [
                        normal_images.head(800),  # 800 normal images
                        other_images.head(200),  # 200 with findings
                    ]
                )
            else:
                final_subset = df_subset.head(1000)
        else:
            final_subset = df_subset.head(1000)

        # Save the usable subset
        os.makedirs("../data/processed", exist_ok=True)
        final_subset.to_csv("../data/processed/usable_subset.csv", index=False)

        print(f"Created usable subset: {len(final_subset)} images")
        print("Saved to: ../data/processed/usable_subset.csv")

        # Show distribution
        if "Finding Labels" in final_subset.columns:
            print("\nFinal subset distribution:")
            print(final_subset["Finding Labels"].value_counts().head(10))

        return final_subset

    return None


if __name__ == "__main__":
    print("Loading CSV...")
    df = pd.read_csv(csv_dir)
    print(f"Total records: {len(df)}")

    print("\nCreating usable subset...")
    df_subset = create_usable_subset(df, img_dir, target_size=1000)
