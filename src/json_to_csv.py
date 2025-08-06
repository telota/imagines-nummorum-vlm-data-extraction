import json
import os
import re

import pandas as pd

# Global variables
ROOT_DIRECTORY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data",
    "example_output",
    "extracted_data",
)
CSV_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data",
    "example_output",
    "coin_data.csv",
)


def natural_sort_key(s):
    """Sort strings containing numbers naturally."""
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


def find_json_files(root_dir):
    """Recursively find all JSON files in the given directory, sorted naturally."""
    json_files = []
    # Get all directories first and sort them naturally
    all_dirs = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        for dirname in dirnames:
            all_dirs.append(os.path.join(dirpath, dirname))

    # Sort directories naturally
    all_dirs.sort(key=natural_sort_key)
    all_dirs.insert(0, root_dir)  # Add root_dir to the beginning

    # Now process each directory in sorted order
    for dirpath in all_dirs:
        filenames = [
            f
            for f in os.listdir(dirpath)
            if f.endswith(".json") and os.path.isfile(os.path.join(dirpath, f))
        ]
        # Sort filenames naturally
        filenames.sort(key=natural_sort_key)
        for filename in filenames:
            json_files.append(os.path.join(dirpath, filename))

    return json_files


def parse_json_file(json_path):
    """Parse a single JSON file and extract relevant fields."""
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Extract basic information
        file_info = {
            "file_name": os.path.basename(json_path),
            "file_path": json_path,
            "image_path_original": data.get("image_path_original", ""),
            "image_type": data.get("image_type", ""),
            "handwritten_content": data.get("handwritten_content", ""),
            "status": data.get("status", ""),
            "error_message": data.get("error_message", ""),
        }

        # Handle form_data which might be null
        form_data = data.get("data", {}).get("form_data", None)

        # Set default value for number of coins
        file_info["num_coins"] = 0

        # Process form_data if it exists
        if form_data:
            # Extract card fields (coin metadata)
            card_fields = form_data.get("card_fields", {}) or {}
            for key, value in card_fields.items():
                file_info[f"card_{key}"] = value

            # Extract all coins
            coins = form_data.get("coins", []) or []
            file_info["num_coins"] = len(coins)

            # Add details for each coin
            for i, coin in enumerate(coins, 1):
                file_info[f"coin{i}_description"] = coin.get("description", "")
                file_info[f"coin{i}_id"] = coin.get("id", "")

                # Optionally include bounding box information
                bbox = coin.get("bounding_box", {})
                file_info[f"coin{i}_bbox_x"] = bbox.get("x", "")
                file_info[f"coin{i}_bbox_y"] = bbox.get("y", "")
                file_info[f"coin{i}_bbox_width"] = bbox.get("width", "")
                file_info[f"coin{i}_bbox_height"] = bbox.get("height", "")

        # Add extracted images information
        file_info["images_extracted"] = ", ".join(data.get("images_extracted", []))

        # Add ocr_results information if available
        ocr_results = data.get("data", {}).get("ocr_results", [])
        file_info["has_ocr_results"] = len(ocr_results) > 0

        return file_info
    except Exception as e:
        print(f"Error processing {json_path}: {str(e)}")
        return None


def main():
    """Process all JSON files and create a DataFrame."""
    global ROOT_DIRECTORY, CSV_OUTPUT

    print(f"Searching for JSON files in {ROOT_DIRECTORY}...")
    json_files = find_json_files(ROOT_DIRECTORY)
    print(f"Found {len(json_files)} JSON files.")

    # Parse each JSON file
    data_rows = []
    for json_file in json_files:
        row_data = parse_json_file(json_file)
        if row_data:
            data_rows.append(row_data)

    # Create DataFrame
    if data_rows:
        df = pd.DataFrame(data_rows)

        # Save to CSV
        df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8")
        print(f"Data saved to {CSV_OUTPUT}")
        print(f"DataFrame shape: {df.shape}")
        return df
    else:
        print("No valid data found.")
        return None


if __name__ == "__main__":
    df = main()

    # Display sample of the data if available
    if df is not None and not df.empty:
        print("\nSample of the data:")
        print(df.head())
