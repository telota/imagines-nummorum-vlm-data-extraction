# API Reference - Imagines Nummorum VLM Data Extraction

## Module Overview

The system consists of three main modules:

1. **`coin_card_information_extraction.py`** - Main processing pipeline
2. **`json_to_csv.py`** - Data conversion utilities
3. **`validate_data.py`** - Processing validation tools

## Main Processing Module

### Core Functions

#### `load_model_and_processor(model_name, cache_dir)`

Loads the Qwen2.5-VL model and processor for image analysis.

**Parameters:**

- `model_name` (str): Hugging Face model identifier
  - Example: `"Qwen/Qwen2.5-VL-32B-Instruct"`
- `cache_dir` (str): Directory to cache downloaded models

**Returns:**

- `tuple`: (model, processor, device)

**Usage:**

```python
model, processor, device = load_model_and_processor(
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "./model_cache"
)
```

#### `process_single_image_multi_stage(image_full_path, output_json_dir_for_image, model, processor, device, input_base_dir)`

Processes a single image through the complete multi-stage pipeline.

**Parameters:**

- `image_full_path` (str): Absolute path to the input image
- `output_json_dir_for_image` (str): Directory for output JSON and extracted images
- `model`: Loaded Qwen2.5-VL model
- `processor`: Loaded model processor
- `device` (str): Processing device ("cuda" or "cpu")
- `input_base_dir` (str): Base input directory for relative path calculation

**Returns:**

- `dict`: Processing results with status and extracted data

**Example Result:**

```python
{
    "image_path_original": "/path/to/image.jpg",
    "image_type": "form",
    "handwritten_content": True,
    "data": {...},
    "images_extracted": ["coin1.png"],
    "status": "success",
    "error_message": None
}
```

#### `batch_process_images_multi_stage(input_dir, output_dir, model_name, cache_dir)`

Processes all images in a directory structure.

**Parameters:**

- `input_dir` (str): Input directory containing images
- `output_dir` (str): Output directory for results
- `model_name` (str): Hugging Face model identifier
- `cache_dir` (str): Model cache directory

**Returns:**

- `None`: Results are written to files

**Usage:**

```python
batch_process_images_multi_stage(
    input_dir="./input_images",
    output_dir="./output_results",
    model_name="Qwen/Qwen2.5-VL-32B-Instruct",
    cache_dir="./model_cache"
)
```

### Helper Functions

#### `_call_qwen_vl_with_retry(...)`

Internal function for making robust API calls to the VLM with retry logic.

**Parameters:**

- `image_path` (str): Path to image file
- `prompt_text` (str): Prompt for the model
- `model`: Loaded model instance
- `processor`: Model processor
- `device` (str): Processing device
- `validation_fn` (callable, optional): JSON validation function
- `schema_to_validate_jsonschema` (dict, optional): JSON schema for validation
- `max_tokens` (int): Maximum tokens in response
- `stage_name` (str): Stage identifier for logging

**Returns:**

- `tuple`: (parsed_json, error_message)

#### `_perform_qwen_text_extraction(image_full_path, model, processor, device, stage_name_suffix="")`

Extracts text content using Qwen-VL model.

**Parameters:**

- `image_full_path` (str): Path to image
- `model`: Loaded model
- `processor`: Model processor
- `device` (str): Processing device
- `stage_name_suffix` (str): Additional identifier for logging

**Returns:**

- `dict`: OCR result with content and status

#### `_ocr_with_tesseract(image_path)`

Performs OCR using Tesseract engine.

**Parameters:**

- `image_path` (str): Path to image file

**Returns:**

- `dict`: OCR result in hOCR format

**Example Result:**

```python
{
    "source": "tesseract",
    "type": "hocr",
    "content": "<html>...hOCR XML...</html>",
    "status": "success",
    "error_message": None
}
```

### Coin Extraction Functions

#### `_find_optimal_margin(original_image, bbox, initial_margin=20, max_margin=100, increment=5)`

Finds optimal cropping margin for coin images.

**Parameters:**

- `original_image` (PIL.Image): Source image
- `bbox` (dict): Bounding box with x, y, width, height
- `initial_margin` (int): Starting margin size in pixels
- `max_margin` (int): Maximum margin to try
- `increment` (int): Margin increment step

**Returns:**

- `int`: Optimal margin size in pixels

#### `_is_edge_uniform(image, edge_width=5, similarity_threshold=30, uniformity_threshold=0.85)`

Checks if image edges are uniform (useful for background detection).

**Parameters:**

- `image` (PIL.Image): Image to analyze
- `edge_width` (int): Width of edge band to check
- `similarity_threshold` (int): RGB color similarity threshold
- `uniformity_threshold` (float): Required uniformity percentage (0.0-1.0)

**Returns:**

- `bool`: True if edges are uniform

#### `_is_color_similar(color1, color2, threshold=30)`

Determines if two RGB colors are similar.

**Parameters:**

- `color1` (tuple): RGB color tuple (r, g, b)
- `color2` (tuple): RGB color tuple (r, g, b)
- `threshold` (int): Maximum difference to consider similar

**Returns:**

- `bool`: True if colors are similar

### Validation Functions

#### `_validate_classification_json(data)`

Validates JSON output from image classification stage.

**Parameters:**

- `data` (dict): JSON data to validate

**Returns:**

- `None`: Raises exception if invalid

**Expected Format:**

```python
{
    "image_type": "form|text_page|empty_page",
    "handwritten_content": True|False
}
```

#### `_validate_qwen_text_extraction_json(data)`

Validates JSON output from text extraction stage.

**Parameters:**

- `data` (dict): JSON data to validate

**Expected Format:**

```python
{
    "extracted_text": "text content as string"
}
```

## Configuration Constants

### Model Configuration

```python
MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"  # Default model
MODEL_CACHE_DIR = "./model_cache"             # Cache directory
```

### Processing Parameters

```python
MAX_JSON_RETRIES = 3                          # Retry attempts
RETRY_DELAY_SECONDS = 1                       # Delay between retries
OVERWRITE_EXISTING_OUTPUT = False             # Skip existing files
```

### OCR Configuration

```python
OCR_STRATEGY_FOR_TEXT_PAGES = "both"          # OCR strategy
TESSERACT_CMD = None                          # Tesseract executable path
```

### Cropping Parameters

```python
CROP_MARGIN_PIXELS = 40                       # Default crop margin
INITIAL_CROP_MARGIN = 20                      # Starting margin
MAX_CROP_MARGIN = 100                         # Maximum margin
MARGIN_INCREMENT = 5                          # Margin increment
EDGE_CHECK_WIDTH = 5                          # Edge analysis width
COLOR_SIMILARITY_THRESHOLD = 30               # Color similarity
EDGE_UNIFORMITY_THRESHOLD = 0.85             # Edge uniformity
DEBUG_VISUALIZATION = True                    # Save debug images
```

### Schema Definitions

#### Form Data Schema

```python
FORM_DATA_VALIDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "coins": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "description": {"type": "string"},
                    "bounding_box": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "integer", "minimum": 0},
                            "y": {"type": "integer", "minimum": 0},
                            "width": {"type": "integer", "minimum": 1},
                            "height": {"type": "integer", "minimum": 1}
                        },
                        "required": ["x", "y", "width", "height"]
                    }
                },
                "required": ["id", "description", "bounding_box"]
            }
        },
        "card_fields": {
            "type": "object",
            "properties": {
                "Atelier": {"type": ["string", "null"]},
                "Date": {"type": ["string", "null"]},
                "Faussaire": {"type": ["string", "null"]},
                "Provenance": {"type": ["string", "null"]},
                "Poids": {"type": ["string", "null"]},
                "Diamètre": {"type": ["string", "null"]},
                "Orientation des axes": {"type": ["string", "null"]},
                "Métal": {"type": ["string", "null"]},
                "Technique": {"type": ["string", "null"]},
                "Publication": {"type": ["string", "null"]},
                "Négatifs": {"type": ["string", "null"]},
                "Remarques": {"type": ["string", "null"]}
            }
        }
    }
}
```

### Prompt Templates

#### Classification Prompt

```python
PROMPT_CLASSIFICATION_HANDWRITING = """Analyze the provided image.
1. Classify the image content type. The type must be one of: "form", "text_page", "empty_page".
2. Determine if the image contains any handwritten text. This should be a boolean value (true or false).
Return your response ONLY as a single JSON object with two keys: "image_type" (string) and "handwritten_content" (boolean).
Example: {"image_type": "form", "handwritten_content": true}"""
```

#### Text Extraction Prompt

```python
PROMPT_TEXT_EXTRACTION_QWEN = """Extract all visible text from this image.
Return your response ONLY as a single JSON object with a single key: "extracted_text" (string).
Example: {"extracted_text": "This is the content of the page..."}"""
```

#### Form Extraction Prompt

```python
PROMPT_FORM_EXTRACTION_TEMPLATE = f"""Analyze the provided image. It might be a form or contain structured elements like coins and an information card.
Identify each coin, provide a brief description, and its bounding box (x, y, width, height where x,y is top-left).
Extract all relevant fields from the information card or form.
Return your response ONLY as a single JSON object strictly adhering to the schema provided below.
If a field is not present, use an empty string "" or null for string fields. Ensure bounding box values are positive integers.

JSON Schema for form data:
{json.dumps(FORM_DATA_VALIDATION_SCHEMA, indent=2)}

Output only the JSON object containing the form data."""
```

## JSON to CSV Converter

### `json_to_csv.py`

#### `find_json_files(root_dir)`

Recursively finds all JSON files in a directory with natural sorting.

**Parameters:**

- `root_dir` (str): Root directory to search

**Returns:**

- `list`: Sorted list of JSON file paths

#### `parse_json_file(json_path)`

Parses a single JSON file and extracts relevant fields for CSV export.

**Parameters:**

- `json_path` (str): Path to JSON file

**Returns:**

- `dict`: Flattened data dictionary or None if error

**Extracted Fields:**

- Basic info: file_name, file_path, image_path_original
- Classification: image_type, handwritten_content
- Status: status, error_message, num_coins
- Card fields: card_Atelier, card_Date, etc.
- Coin data: coin1*description, coin1_id, coin1_bbox*\*
- Additional: images_extracted, has_ocr_results

#### `natural_sort_key(s)`

Generates sort key for natural string sorting (handles numbers correctly).

**Parameters:**

- `s` (str): String to generate sort key for

**Returns:**

- `list`: Sort key components

#### `main()`

Main function that processes all JSON files and creates CSV output.

**Returns:**

- `pandas.DataFrame`: Processed data or None if no data

### Configuration Variables

```python
ROOT_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
CSV_OUTPUT = os.path.join(ROOT_DIRECTORY, "coin_data.csv")
```

## Validation Module

### `validate_data.py`

#### `verify_processing_status(images_dir, output_dir)`

Verifies that all images have been processed successfully.

**Parameters:**

- `images_dir` (str): Directory containing original images
- `output_dir` (str): Directory containing processing results

**Returns:**

- `dict`: Verification results with statistics and error lists

**Result Structure:**

```python
{
    "total_images": 100,
    "processed_successfully": 95,
    "missing_json": [{"image": "path", "expected_json": "path"}],
    "failed_processing": [{"image": "path", "status": "failed", "error": "msg"}],
    "json_parse_errors": [{"json": "path", "error": "msg"}]
}
```

#### `print_results(results)`

Prints verification results in a readable format.

**Parameters:**

- `results` (dict): Results from `verify_processing_status()`

**Returns:**

- `None`: Prints formatted output to console

### Configuration Variables

```python
images_directory = "/path/to/images"      # Set your images directory
output_directory = "/path/to/output"      # Set your output directory
```

## Error Handling

### Exception Types

The system uses standard Python exceptions with descriptive messages:

- `FileNotFoundError`: Missing input files or directories
- `json.JSONDecodeError`: Invalid JSON in model responses
- `jsonschema.ValidationError`: Schema validation failures
- `PIL.UnidentifiedImageError`: Corrupted or invalid image files
- `torch.cuda.OutOfMemoryError`: GPU memory exhaustion
- `pytesseract.TesseractError`: Tesseract OCR failures

### Error Response Format

All functions return error information in a consistent format:

```python
{
    "status": "failed",
    "error_message": "Detailed error description",
    "error_type": "classification_failed|form_extraction_failed|text_extraction_failed"
}
```

## Custom Extensions

### Adding New Image Types

1. **Extend classification prompt**:

```python
PROMPT_CLASSIFICATION_HANDWRITING = """...
The type must be one of: "form", "text_page", "empty_page", "new_type".
..."""
```

2. **Add validation logic**:

```python
def _validate_classification_json(data):
    # ... existing code ...
    if data["image_type"] not in ["form", "text_page", "empty_page", "new_type"]:
        raise ValueError("Invalid image type")
```

3. **Implement processing logic**:

```python
def process_single_image_multi_stage(...):
    # ... existing code ...
    elif classification_result["image_type"] == "new_type":
        # Add custom processing logic
        pass
```

### Custom Card Fields

1. **Modify schema**:

```python
FORM_DATA_VALIDATION_SCHEMA["properties"]["card_fields"]["properties"]["new_field"] = {
    "type": ["string", "null"]
}
```

2. **Update extraction prompt**:

```python
PROMPT_FORM_EXTRACTION_TEMPLATE = """...
Extract fields including: Atelier, Date, new_field, ...
"""
```

3. **Update CSV converter**:

```python
def parse_json_file(json_path):
    # ... existing code ...
    file_info[f"card_new_field"] = card_fields.get("new_field", "")
```

### Custom OCR Strategies

1. **Add new strategy option**:

```python
OCR_STRATEGY_OPTIONS = [
    "tesseract_hocr_only",
    "qwen_text_only",
    "both",
    "tesseract_then_qwen_fallback",
    "qwen_then_tesseract_fallback",
    "custom_strategy"  # New option
]
```

2. **Implement strategy logic**:

```python
def _extract_text_from_page(image_path, strategy, model, processor, device):
    if strategy == "custom_strategy":
        # Implement custom OCR logic
        return custom_ocr_function(image_path)
    # ... existing code ...
```

## Performance Monitoring

### Memory Usage Tracking

```python
import psutil
import torch

def monitor_resources():
    """Monitor system resources during processing."""
    process = psutil.Process()
    return {
        "memory_percent": process.memory_percent(),
        "memory_mb": process.memory_info().rss / 1024 / 1024,
        "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
        "gpu_memory_cached": torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
    }
```

### Processing Time Tracking

```python
import time
from functools import wraps

def time_function(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper
```

This API reference provides comprehensive documentation for all functions, classes, and configuration options in the system. Use it as a guide for customization and extension of the pipeline functionality.
