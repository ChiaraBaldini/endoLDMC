from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def create_datalist(image_dir, mask_dir, caption_dir, extensions=("*.png", "*.jpg", "*.JPG", "*.bmp")):
    """
    Create a data list with paths to images, masks, and captions.

    Args:
        image_dir: Directory containing images.
        mask_dir: Directory containing masks.
        caption_dir: Directory containing captions.
        extensions: Tuple with supported image file extensions.

    Returns:
        DataFrame with paths to images, masks, and captions.
    """
    data_list = []

    # Search for all images with the given extensions
    image_paths = []
    for ext in extensions:
        image_paths.extend(image_dir.glob(ext))

    # For each image, check if corresponding mask and caption exist
    for image_path in sorted(image_paths):
        mask_path = mask_dir / image_path.name
        label_path = caption_dir / (image_path.stem + ".txt")

        if mask_path.exists() and label_path.exists():
            # Read the caption content from the text file
            with open(label_path, 'r', encoding='utf-8') as f:
                caption = "".join(f.readlines()).strip()
            print(str(caption))  # Print the caption (for debugging)

            # Add the sample to the data list
            data_list.append({
                "filename": image_path.name,
                "image": str(image_path),
                "mask": str(mask_path),
                "caption": str(caption)
            })

    # Return a DataFrame with all valid samples
    return pd.DataFrame(data_list)


def split_data(image_paths, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split the data into train, validation, and test sets based on given proportions.

    Args:
        image_paths: List of image paths.
        train_ratio: Proportion for training set.
        val_ratio: Proportion for validation set.
        test_ratio: Proportion for test set.

    Returns:
        Tuple with lists of image paths for train, validation, and test sets.
    """
    # Ensure the ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "sum of train, val and test ratios must be = 1."

    # First, split into train and temp (val+test)
    train_paths, temp_paths = train_test_split(image_paths, train_size=train_ratio, random_state=42)
    # Then, split temp into validation and test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_paths, test_paths = train_test_split(temp_paths, train_size=val_size, random_state=42)

    return train_paths, val_paths, test_paths


def main():
    # Set the output directory for the TSV files
    output_dir = Path("/project/outputs/ids_lar")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set the directories for images, masks, and captions
    image_dir = Path("/project/data/imgs")
    mask_dir = Path("/project/data/masks")
    caption_dir = Path("/project/data/captions")

    # Create the data list DataFrame
    data_df = create_datalist(image_dir, mask_dir, caption_dir)

    # Split the data into train, validation, and test sets
    train_paths, val_paths, test_paths = split_data(data_df['image'].tolist(), train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    # Filter the DataFrame for each split
    train_df = data_df[data_df['image'].isin(train_paths)]
    val_df = data_df[data_df['image'].isin(val_paths)]
    test_df = data_df[data_df['image'].isin(test_paths)]

    # Save each split to a TSV file
    train_df.to_csv(output_dir / "train.tsv", index=False, sep="\t")
    val_df.to_csv(output_dir / "validation.tsv", index=False, sep="\t")
    test_df.to_csv(output_dir / "test.tsv", index=False, sep="\t")

    print(f"✅ TSV file created and saved in: {output_dir}")


if __name__ == "__main__":
    main()
