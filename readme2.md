## Qwen2.5-VL Multi-Stage Image Analysis Pipeline

This script provides an advanced pipeline for analyzing images using the Qwen2.5-VL vision-language model. It processes a directory of images, performs a multi-stage analysis on each, and outputs structured JSON data.

**Current Date:** May 16, 2025

### Features:

1.  **Batch Processing**: Processes all images within a specified input directory (including subdirectories).
2.  **Multi-Stage Analysis per Image**:
    * **Stage 1: Classification & Handwriting Detection**:
        * Classifies the image as "form", "text\_page", or "empty\_page".
        * Detects if the image contains handwritten content (boolean).
    * **Stage 2: Conditional Content Extraction**:
        * **Empty Pages**: No further extraction.
        * **Text Pages**: Extracts textual content using a configurable OCR strategy (Tesseract for hOCR XML, Qwen-VL for plain text, or both).
        * **Forms**: Extracts detailed structured data (e.g., information about coins, card fields) based on a predefined JSON schema.
3.  **Configurable OCR Strategy**: For text pages, choose to use:
    * Tesseract OCR (outputs hOCR XML).
    * Qwen-VL's built-in text recognition (outputs plain text).
    * A combination with fallback options, or both.
4.  **JSON Output**:
    * For each input image, a corresponding JSON file is generated in an output directory, mirroring the input's folder structure.
    * The JSON includes the image classification, handwriting detection status, extracted data (which varies by image type), and paths to any extracted sub-images.
5.  **Coin Image Cropping**: For "form" type images, if coins and their bounding boxes are identified, these coin regions are cropped from the original image and saved as separate image files.
6.  **Retry Mechanism**: Includes a retry mechanism for Qwen-VL API calls if initial JSON parsing or validation fails.
7.  **Progress Tracking**: Uses `tqdm` to display a progress bar during batch processing.
8.  **Efficient Model Loading**: The Qwen-VL model and processor are loaded only once at the beginning of the batch process.

### Prerequisites:

1.  **Python Environment**: Python 3.8 or higher recommended.
2.  **Qwen2.5-VL Model Access**: The script uses `Qwen/Qwen2.5-VL-32B-Instruct` by default. Ensure you have network access to download it from Hugging Face Hub upon first run, or have it cached locally. This model is large (approx. 60GB+) and requires significant RAM and potentially GPU resources.
3.  **Tesseract OCR Engine (Optional, for Tesseract OCR strategy)**:
    * If you plan to use Tesseract OCR, you must install the Tesseract OCR engine on your system.
        * **Linux (Ubuntu/Debian)**: `sudo apt update && sudo apt install tesseract-ocr tesseract-ocr-eng` (Install `tesseract-ocr-[langcode]` for other languages).
        * **macOS**: `brew install tesseract tesseract-lang`
        * **Windows**: Download and run the installer from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki). Ensure Tesseract is added to your system's PATH environment variable during installation.
    * Verify Tesseract installation by running `tesseract --version` in your terminal.

### Setup:

1.  **Clone or Download Script**: Place the Python script (e.g., `qwen_pipeline.py` or in a Jupyter Notebook cell) in your project directory.
2.  **Create Directories**:
    * Create an input directory for your images (e.g., `./input_images_for_qwen_final_v3/`). You can organize images into subfolders within this directory.
    * The script will automatically create the output directory for JSON results and cropped images if it doesn't exist (e.g., `./qwen_json_output_final_results_v3/`).
3.  **Install Python Dependencies**:
    It is recommended to use a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```
    If you don't have a `requirements.txt` file, install the packages listed at the beginning of the script (or in the `requirements.txt` content provided above).

### Configuration:

Open the script and modify the following global variables/constants at the beginning or within the `if __name__ == '__main__':` block:

* **File Paths**:
    * `INPUT_IMAGE_DIRECTORY`: Path to your main folder containing images to process.
    * `OUTPUT_JSON_DIRECTORY`: Path where output JSON files and cropped images will be saved.
* **Model Configuration**:
    * `MODEL_ID`: The Hugging Face model ID for Qwen2.5-VL (e.g., `"Qwen/Qwen2.5-VL-32B-Instruct"` or `"Qwen/Qwen2.5-VL-7B-Instruct"` for a smaller version).
    * `MODEL_CACHE_DIR`: Directory to cache the downloaded Hugging Face model.
* **Processing Behavior**:
    * `MAX_JSON_RETRIES`: Number of times to retry Qwen-VL calls if JSON parsing/validation fails (default: 3).
    * `RETRY_DELAY_SECONDS`: Delay in seconds between retries (default: 1).
* **OCR Strategy for Text Pages**:
    * `OCR_STRATEGY_FOR_TEXT_PAGES`: Choose the OCR method for images classified as "text\_page". Options:
        * `"tesseract_hocr_only"`: Only Tesseract for hOCR output.
        * `"qwen_text_only"`: Only Qwen-VL for plain text output.
        * `"tesseract_then_qwen_fallback"`: Try Tesseract first; if it fails, try Qwen-VL.
        * `"qwen_then_tesseract_fallback"`: Try Qwen-VL first; if it fails, try Tesseract.
        * `"both"`: Run both Tesseract and Qwen-VL independently.
* **Tesseract Path (If Needed)**:
    * If `pytesseract` cannot find your Tesseract installation automatically, you might need to uncomment and set the path explicitly in the script:
        ```python
        # import pytesseract
        # pytesseract.pytesseract.tesseract_cmd = r'/path/to/your/tesseract' # e.g., /usr/bin/tesseract or C:\Program Files\Tesseract-OCR\tesseract.exe
        ```

### Running the Script:

1.  **Activate Virtual Environment** (if used).
2.  **Ensure Configuration**: Double-check that `INPUT_IMAGE_DIRECTORY` and other configurations in the script are set correctly.
3.  **Execute the Script**:
    * If it's a `.py` file: `python your_script_name.py`
    * If it's in a Jupyter Notebook: Run the cell containing the script.

The script will then:
* Load the Qwen-VL model (downloading it if run for the first time).
* Scan the `INPUT_IMAGE_DIRECTORY` for images.
* Process each image using the multi-stage pipeline, with a `tqdm` progress bar indicating progress.
* Save the resulting JSON file and any cropped images to the `OUTPUT_JSON_DIRECTORY`, maintaining the original folder structure.
* Print a summary of the batch processing at the end.

### Output Structure:

For each input image (e.g., `input_dir/subdir/my_image.jpg`), the script generates:

1.  **JSON File**: `output_dir/subdir/my_image.json`
    The JSON file contains:
    * `image_path_original`: Original path of the processed image.
    * `image_type`: Classification result ("form", "text\_page", "empty\_page").
    * `handwritten_content`: Boolean indicating presence of handwriting.
    * `data`: An object whose structure depends on `image_type`:
        * For "empty\_page": `null` or a minimal message.
        * For "text\_page": Contains an `ocr_results` array with details from Tesseract (hOCR) and/or Qwen-VL (plain text) based on the chosen strategy. Each result includes `source`, `type`, `content`, `status`, and `error_message`.
        * For "form": Contains the detailed structured data as defined by `FORM_DATA_VALIDATION_SCHEMA` (e.g., `coins` array, `card_fields` object).
    * `images_extracted`: An array of relative paths to cropped images (e.g., for coins), stored in a subdirectory. Example: `["my_image_extracted_images/coin_1_id.png"]`.
    * `status`: Overall processing status for the image (e.g., "success", "classification\_failed", "form\_extraction\_failed").
    * `error_message`: Details if an error occurred.

2.  **Cropped Images (for "form" type)**:
    * If coins are extracted from a form, cropped images of these coins are saved in:
        `output_dir/subdir/my_image_extracted_images/`
    * Filenames will be based on the coin ID (e.g., `coin1_obverse.png`).

### Troubleshooting:

* **Tesseract Not Found**: Ensure Tesseract OCR engine is installed correctly and either in your system's PATH or `pytesseract.pytesseract.tesseract_cmd` is set correctly in the script.
* **Model Download Issues**: Check your internet connection. Large model downloads can take time.
* **Out of Memory Errors**: The Qwen2.5-VL 32B model is very large. If you encounter memory issues, consider:
    * Using a machine with more RAM/VRAM.
    * Trying a smaller version of the model (e.g., `Qwen/Qwen2.5-VL-7B-Instruct`).
    * Closing other resource-intensive applications.
* **JSON Errors from Model**: The script includes retries. If errors persist, the prompts might need further refinement for your specific image types, or the model may struggle with particularly complex/low-quality images. Check the `error_message` in the output JSON and `tqdm` logs for clues.
* **Pillow/Image Errors**: Ensure your image files are not corrupted and are in common formats (JPG, PNG, etc.).