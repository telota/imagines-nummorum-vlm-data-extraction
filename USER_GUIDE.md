# User Guide - Imagines Nummorum VLM Data Extraction

## Quick Start

### Basic Usage

1. **Prepare your images**: Place images in an input directory
2. **Configure the script**: Set input/output paths
3. **Run processing**: Execute the main script
4. **Review results**: Check JSON outputs and extracted images

### Simple Example

```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Run processing
python src/coin_card_information_extraction.py
```

## Step-by-Step Tutorial

### Step 1: Organize Your Images

Create a well-organized input directory structure:

```
input_images/
├── collection_A/
│   ├── coin_card_001.jpg
│   ├── coin_card_002.jpg
│   └── text_page_001.jpg
├── collection_B/
│   ├── form_001.png
│   └── form_002.png
└── single_items/
    ├── item_001.tiff
    └── item_002.jpg
```

**Supported formats**: JPG, JPEG, PNG, TIFF, TIF, BMP

### Step 2: Configure Processing Parameters

Edit the main script (`src/coin_card_information_extraction.py`) to set your paths:

```python
# Essential configuration
INPUT_IMAGE_DIRECTORY = "C:/path/to/your/input_images"
OUTPUT_JSON_DIRECTORY = "C:/path/to/your/output_results"

# Model configuration
MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"  # or 7B for smaller model
MODEL_CACHE_DIR = "C:/path/to/model_cache"

# OCR strategy for text pages
OCR_STRATEGY_FOR_TEXT_PAGES = "both"  # tesseract_hocr_only, qwen_text_only, both
```

### Step 3: Run the Processing Pipeline

Execute the main script:

```bash
python src/coin_card_information_extraction.py
```

You'll see progress output like:

```
Loading Qwen-VL model...
Set Hugging Face cache directory to: C:\model_cache
Using device: cuda
Processing images: 100%|████████████| 50/50 [02:15<00:00,  2.70s/it]
Batch processing completed.
Total images processed: 50
```

### Step 4: Review Results

The output directory will mirror your input structure:

```
output_results/
├── collection_A/
│   ├── coin_card_001.json
│   ├── coin_card_001_extracted_images/
│   │   ├── coin1_obverse.png
│   │   └── coin1_reverse.png
│   ├── coin_card_002.json
│   └── text_page_001.json
├── collection_B/
│   ├── form_001.json
│   └── form_002.json
└── single_items/
    ├── item_001.json
    └── item_002.json
```

## Understanding Output Files

### JSON Structure

Each processed image generates a JSON file with the following structure:

```json
{
  "image_path_original": "/path/to/original/image.jpg",
  "image_type": "form",
  "handwritten_content": true,
  "data": {
    "form_data": {
      "coins": [
        {
          "id": "coin_1",
          "description": "Ancient Greek silver tetradrachm",
          "bounding_box": {
            "x": 150,
            "y": 100,
            "width": 200,
            "height": 200
          }
        }
      ],
      "card_fields": {
        "Atelier": "Athens",
        "Date": "440-430 BC",
        "Métal": "Silver",
        "Poids": "17.2g",
        "Diamètre": "24mm"
      }
    }
  },
  "images_extracted": ["coin_card_001_extracted_images/coin_1.png"],
  "status": "success",
  "error_message": null
}
```

### Image Types and Their Outputs

#### 1. Form Images

- **Contains**: Coin images and metadata cards
- **Output**: Structured data + cropped coin images
- **Use case**: Museum catalog cards, documentation forms

**Example output**:

```json
{
  "image_type": "form",
  "data": {
    "form_data": {
      "coins": [...],
      "card_fields": {...}
    }
  },
  "images_extracted": ["coin1.png", "coin2.png"]
}
```

#### 2. Text Pages

- **Contains**: Textual content, manuscripts, printed pages
- **Output**: Extracted text using OCR
- **Use case**: Historical documents, catalog pages

**Example output**:

```json
{
  "image_type": "text_page",
  "data": {
    "ocr_results": [
      {
        "source": "tesseract",
        "type": "hocr",
        "content": "<xml>hOCR formatted text</xml>",
        "status": "success"
      },
      {
        "source": "qwen_vl",
        "type": "plain_text",
        "content": "Plain text extraction",
        "status": "success"
      }
    ]
  }
}
```

#### 3. Empty Pages

- **Contains**: Blank or nearly empty pages
- **Output**: Minimal processing
- **Use case**: Separator pages, blank forms

## Advanced Usage

### Batch Processing Large Collections

For processing thousands of images:

1. **Monitor system resources**:

```bash
# Linux/macOS - monitor memory usage
watch -n 5 free -h
htop

# Windows - Task Manager or PowerShell
Get-Process | Sort-Object WorkingSet -Descending
```

2. **Process in chunks** if needed:

```python
# Modify the script to process subdirectories separately
import os
subdirs = [d for d in os.listdir(INPUT_IMAGE_DIRECTORY)
          if os.path.isdir(os.path.join(INPUT_IMAGE_DIRECTORY, d))]

for subdir in subdirs:
    input_path = os.path.join(INPUT_IMAGE_DIRECTORY, subdir)
    output_path = os.path.join(OUTPUT_JSON_DIRECTORY, subdir)
    # Process this subdirectory
```

### Custom OCR Strategies

Choose the appropriate OCR strategy based on your content:

```python
# For multilingual text
OCR_STRATEGY_FOR_TEXT_PAGES = "tesseract_hocr_only"

# For handwritten content
OCR_STRATEGY_FOR_TEXT_PAGES = "qwen_text_only"

# For maximum accuracy
OCR_STRATEGY_FOR_TEXT_PAGES = "both"

# For fallback processing
OCR_STRATEGY_FOR_TEXT_PAGES = "qwen_then_tesseract_fallback"
```

### Optimizing Coin Extraction

Adjust cropping parameters for different image qualities:

```python
# For high-resolution images
CROP_MARGIN_PIXELS = 60
MAX_CROP_MARGIN = 150

# For low-resolution or noisy images
INITIAL_CROP_MARGIN = 30
COLOR_SIMILARITY_THRESHOLD = 40
EDGE_UNIFORMITY_THRESHOLD = 0.75
```

## Data Analysis Workflow

### Convert to CSV for Analysis

After processing, convert JSON results to CSV:

```bash
python src/json_to_csv.py
```

This creates `coin_data.csv` with flattened data suitable for:

- Excel analysis
- Database import
- Statistical analysis
- Machine learning training

### Validate Processing Results

Check processing completeness:

```bash
python src/validate_data.py
```

**Sample validation output**:

```
=== Processing Verification Results ===
Total images found: 1247
Successfully processed: 1189
Success rate: 95.3%

❌ Missing JSON files (12):
  - /path/to/image1.jpg → /path/to/image1.json
  - /path/to/image2.jpg → /path/to/image2.json

❌ Failed processing (46):
  - /path/to/image3.jpg
    Status: classification_failed
    Error: Unable to classify image content
```

## Working with Specific Content Types

### Museum Catalog Cards

Typical workflow for museum documentation:

1. **Scan catalog cards** at 300+ DPI
2. **Organize by collection** or period
3. **Use "both" OCR strategy** for maximum text extraction
4. **Review card_fields extraction** for metadata accuracy
5. **Export to CSV** for database integration

### Archaeological Documentation

For archaeological finds documentation:

1. **Process documentation forms** with embedded photos
2. **Extract coin/artifact images** automatically
3. **Preserve spatial relationships** using bounding boxes
4. **Cross-reference with field notes** using extracted text

### Digital Archive Processing

For large digital collections:

1. **Batch process by date/collection**
2. **Monitor for handwritten annotations**
3. **Validate OCR quality** for searchable archives
4. **Generate metadata catalogs** from card fields

## Quality Assurance

### Reviewing Results

1. **Check success rates**:

```bash
python src/validate_data.py
```

2. **Sample random outputs**:

```bash
# Review a random sample of JSON files
ls output_results/**/*.json | shuf -n 10 | xargs -I {} jq '.status' {}
```

3. **Validate extracted images**:

```python
import os
from PIL import Image

def check_extracted_images(output_dir):
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.png'):
                try:
                    img = Image.open(os.path.join(root, file))
                    if img.size[0] < 50 or img.size[1] < 50:
                        print(f"Small image: {file} - {img.size}")
                except Exception as e:
                    print(f"Corrupted image: {file} - {e}")
```

### Common Quality Issues

1. **Poor OCR results**: Check image quality, adjust OCR strategy
2. **Missing coin extractions**: Verify bounding box detection
3. **Incorrect classifications**: Review problematic images manually
4. **Incomplete metadata**: Check card field extraction accuracy

## Performance Optimization

### Hardware Optimization

1. **Use GPU acceleration**:

```python
# Verify GPU usage
import torch
print(f"Using GPU: {torch.cuda.get_device_name(0)}")
```

2. **Optimize memory usage**:

```python
# Clear cache between batches
torch.cuda.empty_cache()
```

3. **Adjust batch sizes** based on available memory

### Processing Optimization

1. **Pre-filter images** to remove obvious non-content
2. **Use smaller model** for initial classification, larger for extraction
3. **Parallel processing** for independent operations
4. **Resume interrupted processing** using existing JSON checks

## Integration with External Tools

### Database Integration

Export to various database formats:

```python
import pandas as pd
import sqlite3

# Load CSV data
df = pd.read_csv('output/coin_data.csv')

# Export to SQLite
conn = sqlite3.connect('numismatic_data.db')
df.to_sql('coin_records', conn, if_exists='replace', index=False)
```

### Web Interface Integration

Create a simple web viewer:

```python
from flask import Flask, render_template, jsonify
import json
import os

app = Flask(__name__)

@app.route('/api/results')
def get_results():
    results = []
    for root, dirs, files in os.walk('output_results'):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file)) as f:
                    results.append(json.load(f))
    return jsonify(results)
```

### Export Formats

Convert to various formats for different applications:

```python
# Export to Excel with multiple sheets
with pd.ExcelWriter('numismatic_analysis.xlsx') as writer:
    coins_df.to_excel(writer, sheet_name='Coins', index=False)
    metadata_df.to_excel(writer, sheet_name='Metadata', index=False)

# Export to TEI XML for digital humanities
def export_to_tei(data):
    # Convert structured data to TEI format
    pass
```

## Troubleshooting Common Issues

### Processing Failures

1. **Check input image format and quality**
2. **Verify sufficient disk space and memory**
3. **Review error messages in JSON outputs**
4. **Test with single images first**

### Performance Issues

1. **Monitor system resources during processing**
2. **Adjust model size based on available hardware**
3. **Process smaller batches if memory is limited**
4. **Use CPU-only mode if GPU issues occur**

For detailed troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Best Practices

### File Organization

- Use descriptive directory names
- Maintain consistent naming conventions
- Keep original images in separate backup location
- Document your organizational scheme

### Processing Workflow

- Start with small test batches
- Validate results before large-scale processing
- Monitor system resources during long runs
- Keep processing logs for troubleshooting

### Data Management

- Regular backups of processing results
- Version control for configuration changes
- Document any manual corrections or annotations
- Maintain chain of custody for source materials

## Next Steps

After mastering basic usage:

1. Review [Technical Documentation](TECHNICAL_DOCUMENTATION.md) for advanced features
2. Explore [API Reference](API_REFERENCE.md) for customization options
3. Check [Examples](examples/) directory for specific use cases
4. Consider contributing improvements via GitHub
