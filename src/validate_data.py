import os
import json


def verify_processing_status(images_dir, output_dir):
    """
    Verify that all images have been processed and have JSON files with status: success
    """
    results = {
        "total_images": 0,
        "processed_successfully": 0,
        "missing_json": [],
        "failed_processing": [],
        "json_parse_errors": [],
    }

    # Walk through all image files
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            # Check if it's an image file
            if file.lower().endswith(
                (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp")
            ):
                results["total_images"] += 1

                # Get relative path from images_dir
                rel_path = os.path.relpath(root, images_dir)
                image_name = os.path.splitext(file)[0]

                # Construct expected JSON path
                if rel_path == ".":
                    json_path = os.path.join(output_dir, f"{image_name}.json")
                else:
                    json_path = os.path.join(output_dir, rel_path, f"{image_name}.json")

                # Check if JSON file exists
                if not os.path.exists(json_path):
                    results["missing_json"].append(
                        {"image": os.path.join(root, file), "expected_json": json_path}
                    )
                    continue

                # Check JSON content
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    if data.get("status") == "success":
                        results["processed_successfully"] += 1
                    else:
                        results["failed_processing"].append(
                            {
                                "image": os.path.join(root, file),
                                "json": json_path,
                                "status": data.get("status", "unknown"),
                                "error": data.get("error_message"),
                            }
                        )

                except json.JSONDecodeError as e:
                    results["json_parse_errors"].append(
                        {"json": json_path, "error": str(e)}
                    )
                except Exception as e:
                    results["json_parse_errors"].append(
                        {"json": json_path, "error": f"Unexpected error: {str(e)}"}
                    )

    return results


def print_results(results):
    """Print verification results in a readable format"""
    print("=== Processing Verification Results ===")
    print(f"Total images found: {results['total_images']}")
    print(f"Successfully processed: {results['processed_successfully']}")
    print(
        f"Success rate: {results['processed_successfully']/results['total_images']*100:.1f}%"
        if results["total_images"] > 0
        else "No images found"
    )
    print()

    if results["missing_json"]:
        print(f"❌ Missing JSON files ({len(results['missing_json'])}):")
        for item in results["missing_json"]:
            print(f"  - {item['image']} → {item['expected_json']}")
        print()

    if results["failed_processing"]:
        print(f"❌ Failed processing ({len(results['failed_processing'])}):")
        for item in results["failed_processing"]:
            print(f"  - {item['image']}")
            print(f"    Status: {item['status']}")
            if item["error"]:
                print(f"    Error: {item['error']}")
        print()

    if results["json_parse_errors"]:
        print(f"❌ JSON parse errors ({len(results['json_parse_errors'])}):")
        for item in results["json_parse_errors"]:
            print(f"  - {item['json']}: {item['error']}")
        print()

    if not any(
        [
            results["missing_json"],
            results["failed_processing"],
            results["json_parse_errors"],
        ]
    ):
        print("✅ All images have been successfully processed!")


if __name__ == "__main__":
    # Set your directories here
    images_directory = "/mnt/data/projects/imagines_nummorum/coin_cards/scans"  # Replace with your images directory
    output_directory = "/mnt/data/projects/imagines_nummorum/coin_cards/output"  # Replace with your output directory

    # Verify processing
    results = verify_processing_status(images_directory, output_directory)
    print_results(results)

    # Optionally save results to file
    with open("processing_verification.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nDetailed results saved to: processing_verification.json")
