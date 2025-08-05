# Technical Documentation - Imagines Nummorum VLM Data Extraction

## Overview

The Imagines Nummorum VLM data extraction pipeline is a sophisticated computer vision and natural language processing system designed to automatically analyze and extract structured data from historical coin documentation. The system leverages the Qwen2.5-VL vision-language model to perform multi-stage image analysis.

## System Architecture

### Core Components

1. **`coin_card_information_extraction.py`** - Main pipeline script
2. **`json_to_csv.py`** - Data conversion utility
3. **`validate_data.py`** - Processing verification tool

### Data Flow

```
Input Images → Classification → Content Extraction → JSON Output → CSV Conversion
     ↓              ↓                ↓                ↓             ↓
  JPG/PNG    form/text_page/    OCR/Structured     JSON Files   CSV Export
   Files      empty_page         Data             + Cropped     Database
                                                   Images
```

## Multi-Stage Processing Pipeline

### Stage 1: Image Classification & Handwriting Detection

**Purpose**: Categorize images and detect handwritten content

**Process**:

1. Load image using PIL
2. Send to Qwen2.5-VL with classification prompt
3. Return classification: `"form"`, `"text_page"`, or `"empty_page"`
4. Detect handwritten content: `true` or `false`

**JSON Schema**:

```json
{
  "image_type": "form|text_page|empty_page",
  "handwritten_content": true|false
}
```

### Stage 2: Conditional Content Extraction

#### For "empty_page" Images

- No further processing
- Status: `"success"` with minimal data

#### For "text_page" Images

- Configurable OCR strategies:
  - **Tesseract Only**: Outputs hOCR XML format
  - **Qwen-VL Only**: Outputs plain text
  - **Both**: Parallel processing with both methods
  - **Fallback**: Primary method with secondary backup

**OCR Result Schema**:

```json
{
  "ocr_results": [
    {
      "source": "tesseract|qwen_vl",
      "type": "hocr|plain_text",
      "content": "extracted text content",
      "status": "success|failed",
      "error_message": "error details if failed"
    }
  ]
}
```

#### For "form" Images

- Structured data extraction using predefined JSON schema
- Coin detection with bounding boxes
- Card field extraction (metadata)
- Automatic coin image cropping

**Form Data Schema**:

```json
{
  "coins": [
    {
      "id": "coin_identifier",
      "description": "coin description",
      "bounding_box": {
        "x": 0,
        "y": 0,
        "width": 100,
        "height": 100
      }
    }
  ],
  "card_fields": {
    "Atelier": "workshop/mint",
    "Date": "dating information",
    "Faussaire": "counterfeiter info",
    "Provenance": "origin/provenance",
    "Poids": "weight",
    "Diamètre": "diameter",
    "Orientation des axes": "axis orientation",
    "Métal": "metal composition",
    "Technique": "manufacturing technique",
    "Publication": "publication reference",
    "Négatifs": "negative numbers",
    "Remarques": "additional remarks"
  }
}
```

## Advanced Features

### Intelligent Coin Cropping

The system includes sophisticated logic for extracting coin images with optimal margins:

**Margin Detection Algorithm**:

1. Start with initial margin (default: 20px)
2. Check edge uniformity using color similarity analysis
3. Incrementally increase margin if edges show non-uniform background
4. Stop when optimal margin is found or maximum reached

**Parameters**:

- `INITIAL_CROP_MARGIN`: Starting margin size (20px)
- `MAX_CROP_MARGIN`: Maximum margin size (100px)
- `MARGIN_INCREMENT`: Increment step (5px)
- `COLOR_SIMILARITY_THRESHOLD`: RGB distance threshold (30)
- `EDGE_UNIFORMITY_THRESHOLD`: Uniformity percentage (85%)

### Error Handling & Retry Mechanism

**Retry Logic**:

- Maximum retries: 3 attempts (configurable)
- Retry delay: 1 second (configurable)
- JSON validation after each attempt
- Schema validation for structured data

**Error Categories**:

- `classification_failed`: Stage 1 processing errors
- `form_extraction_failed`: Form data extraction errors
- `text_extraction_failed`: OCR processing errors
- `json_parse_error`: JSON parsing/validation errors

### Performance Optimizations

1. **Model Loading**: Single model load per batch process
2. **Memory Management**: Automatic cleanup of large objects
3. **Progress Tracking**: tqdm-based progress bars
4. **Batch Processing**: Efficient directory traversal with natural sorting

## Configuration Parameters

### Model Configuration

```python
MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"  # or 7B for smaller version
MODEL_CACHE_DIR = "./model_cache"
```

### Processing Configuration

```python
MAX_JSON_RETRIES = 3
RETRY_DELAY_SECONDS = 1
OVERWRITE_EXISTING_OUTPUT = False
```

### OCR Configuration

```python
OCR_STRATEGY_FOR_TEXT_PAGES = "both"
TESSERACT_CMD = None  # Auto-detect or specify path
```

### Cropping Configuration

```python
CROP_MARGIN_PIXELS = 40
INITIAL_CROP_MARGIN = 20
MAX_CROP_MARGIN = 100
MARGIN_INCREMENT = 5
EDGE_CHECK_WIDTH = 5
COLOR_SIMILARITY_THRESHOLD = 30
EDGE_UNIFORMITY_THRESHOLD = 0.85
DEBUG_VISUALIZATION = True
```

## Output Structure

### JSON Output Format

```json
{
  "image_path_original": "/path/to/original/image.jpg",
  "image_type": "form|text_page|empty_page",
  "handwritten_content": true|false,
  "data": {
    "form_data": { /* form-specific data */ },
    "ocr_results": [ /* OCR results for text pages */ ]
  },
  "images_extracted": [
    "relative/path/to/coin1.png",
    "relative/path/to/coin2.png"
  ],
  "status": "success|classification_failed|form_extraction_failed|text_extraction_failed",
  "error_message": "detailed error message if failed"
}
```

### Directory Structure

```
output_directory/
├── subfolder1/
│   ├── image1.json
│   ├── image1_extracted_images/
│   │   ├── image1_1.png
│   │   └── image1_2.png
│   └── image2.json
└── subfolder2/
    └── image3.json
```

## Data Processing Tools

### JSON to CSV Converter (`json_to_csv.py`)

**Purpose**: Convert batch processing results to tabular format for analysis

**Features**:

- Natural sorting of files and directories
- Flattened coin data (coin1_description, coin2_description, etc.)
- Card field extraction
- Bounding box coordinates
- Processing status information

**Output Columns**:

- File metadata: `file_name`, `file_path`, `image_path_original`
- Classification: `image_type`, `handwritten_content`
- Processing: `status`, `error_message`, `num_coins`
- Card fields: `card_Atelier`, `card_Date`, etc.
- Coin data: `coin1_description`, `coin1_id`, `coin1_bbox_*`
- Additional: `images_extracted`, `has_ocr_results`

### Processing Validator (`validate_data.py`)

**Purpose**: Verify processing completeness and identify issues

**Validation Checks**:

- Missing JSON files for processed images
- Failed processing status
- JSON parsing errors
- Success rate calculation

**Output Reports**:

- Summary statistics
- Detailed error lists
- Processing verification JSON file

## System Requirements

### Hardware Requirements

- **RAM**: 64GB+ for 32B model, 16GB+ for 7B model
- **Storage**: 100GB+ for model cache
- **GPU**: NVIDIA GPU with 24GB+ VRAM (recommended)
- **CPU**: Multi-core processor for efficient batch processing

### Software Requirements

- **Python**: 3.8+ (3.9+ recommended)
- **CUDA**: 11.8+ for GPU acceleration
- **Tesseract**: 4.0+ for OCR functionality
- **Operating System**: Windows 10+, macOS 10.15+, Ubuntu 18.04+

### Network Requirements

- Internet connection for initial model download (~60GB for 32B model)
- Hugging Face Hub access for model and tokenizer downloads

## Security Considerations

### Data Privacy

- All processing occurs locally
- No data transmitted to external services (except model downloads)
- Configurable cache directories for sensitive environments

### Model Security

- Models downloaded from official Hugging Face repositories
- SHA validation recommended for production environments
- Local caching prevents repeated downloads

## Performance Benchmarks

### Processing Speed (approximate)

- **Classification**: ~2-5 seconds per image
- **Form Extraction**: ~10-30 seconds per image
- **Text OCR**: ~5-15 seconds per image
- **Batch Processing**: Varies by image count and complexity

### Memory Usage

- **32B Model**: ~60GB+ RAM during processing
- **7B Model**: ~15GB+ RAM during processing
- **Image Processing**: ~100MB per high-resolution image

## Extensibility

### Adding New Image Types

1. Extend classification prompt in `PROMPT_CLASSIFICATION_HANDWRITING`
2. Add validation logic in `_validate_classification_json`
3. Implement extraction logic in `process_single_image_multi_stage`

### Custom Card Fields

1. Modify `FORM_DATA_VALIDATION_SCHEMA`
2. Update extraction prompt in `PROMPT_FORM_EXTRACTION_TEMPLATE`
3. Adjust CSV converter in `json_to_csv.py`

### OCR Strategy Extension

1. Add new strategy option in configuration
2. Implement extraction logic in processing pipeline
3. Update result schema and validation

## Version History

- **v1.0** (2025-06): Initial release with Qwen2.5-VL integration
