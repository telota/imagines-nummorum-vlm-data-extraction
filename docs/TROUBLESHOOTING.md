# Troubleshooting Guide - Imagines Nummorum VLM Data Extraction

## Common Issues and Solutions

### Installation Issues

#### 1. PyTorch CUDA Installation Problems

**Problem**: CUDA version mismatch or PyTorch not using GPU

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Solutions**:

```bash
# Check CUDA version
nvidia-smi
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

#### 2. Tesseract OCR Not Found

**Problem**: `pytesseract.TesseractNotFoundError`

**Solutions**:

**Windows**:

```python
import pytesseract
# Set explicit path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Or add to system PATH
# Add C:\Program Files\Tesseract-OCR to PATH environment variable
```

**Linux**:

```bash
# Install Tesseract
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-eng

# Verify installation
tesseract --version
which tesseract
```

**macOS**:

```bash
# Install via Homebrew
brew install tesseract

# If not in PATH
export PATH="/opt/homebrew/bin:$PATH"
```

#### 3. Transformers Model Download Issues

**Problem**: Model download fails or is very slow

**Solutions**:

```bash
# Set proxy if needed
export HF_ENDPOINT=https://hf-mirror.com

# Manual download using git-lfs
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct

# Or use smaller model for testing
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
```

#### 4. Memory Issues During Installation

**Problem**: Out of memory when installing large packages

**Solutions**:

```bash
# Use --no-cache-dir to save memory
pip install --no-cache-dir transformers torch

# Install packages one by one
pip install torch
pip install transformers
pip install accelerate
```

### Runtime Issues

#### 1. Out of Memory Errors

**Problem**: CUDA out of memory or system RAM exhausted

```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions**:

**GPU Memory**:

```python
# Use smaller model
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# Clear CUDA cache
import torch
torch.cuda.empty_cache()

# Use CPU if necessary
device = "cpu"  # Force CPU usage

# Enable gradient checkpointing (if available)
model.gradient_checkpointing_enable()
```

**System Memory**:

```python
# Process images in smaller batches
def process_batch(image_paths, batch_size=5):
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        for image_path in batch:
            process_single_image(image_path)
        # Clear memory between batches
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

**Monitor Memory Usage**:

```python
import psutil
import gc

def monitor_memory():
    process = psutil.Process()
    memory_percent = process.memory_percent()
    print(f"Memory usage: {memory_percent:.1f}%")
    if memory_percent > 80:
        print("Warning: High memory usage")
        gc.collect()
```

#### 2. Model Loading Failures

**Problem**: Model fails to load or takes too long

**Solutions**:

```python
# Check available disk space
import shutil
total, used, free = shutil.disk_usage("/")
print(f"Free space: {free // (1024**3)} GB")

# Load with explicit device mapping
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",  # Automatic device distribution
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)

# Alternative: Load in CPU mode first
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    device_map="cpu"
)
# Then move to GPU if needed
if torch.cuda.is_available():
    model = model.to("cuda")
```

#### 3. JSON Parsing Errors

**Problem**: Model returns invalid JSON or parsing fails

**Solutions**:

````python
# Enhanced JSON cleaning
def clean_and_parse_json(json_str):
    try:
        # Remove common prefixes/suffixes
        json_str = json_str.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]

        # Remove leading/trailing whitespace
        json_str = json_str.strip()

        # Try parsing
        return json.loads(json_str), None
    except json.JSONDecodeError as e:
        # Try to fix common issues
        try:
            # Remove trailing commas
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            return json.loads(json_str), None
        except:
            return None, f"JSON parsing failed: {str(e)}"

# Increase max_tokens if responses are truncated
max_tokens = 2048  # or higher
````

#### 4. JSON Errors from Model

**Problem**: Model returns inconsistent JSON or struggles with specific image types

**Solutions**:

```python
# Enhanced prompt refinement for specific use cases
def get_enhanced_classification_prompt():
    return """Analyze this image very carefully.

For FORMS: Look for structured layouts, catalog cards, forms with fields, coin images with labels or metadata.
For TEXT_PAGES: Look for primarily textual content, manuscripts, printed documents, continuous text.
For EMPTY_PAGES: Look for blank/nearly blank pages, separator pages, minimal content.

Return ONLY valid JSON: {"image_type": "form|text_page|empty_page", "handwritten_content": true|false}"""

# Check output JSON structure validity
def validate_and_repair_json(json_data, expected_schema):
    # Add validation logic
    if "image_type" not in json_data:
        return None, "Missing required field: image_type"

    valid_types = ["form", "text_page", "empty_page"]
    if json_data["image_type"] not in valid_types:
        return None, f"Invalid image_type: {json_data['image_type']}"

    return json_data, None

# Monitor and log problematic images
def log_processing_issues(image_path, error_message):
    with open("processing_issues.log", "a") as f:
        f.write(f"{datetime.now()}: {image_path} - {error_message}\n")
```

**Additional Steps:**

- Check the `error_message` field in output JSON files
- Review `tqdm` progress logs for patterns in failures
- For persistent issues, consider manual review of problematic images
- Adjust prompts based on your specific image types and content

#### 4. Image Processing Errors

**Problem**: PIL errors when opening or processing images

**Solutions**:

```python
def safe_image_open(image_path):
    try:
        # Verify file exists and is readable
        if not os.path.exists(image_path):
            return None, f"File not found: {image_path}"

        # Check file size
        file_size = os.path.getsize(image_path)
        if file_size == 0:
            return None, f"Empty file: {image_path}"

        # Try to open
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Check reasonable dimensions
            if img.size[0] < 10 or img.size[1] < 10:
                return None, f"Image too small: {img.size}"

            # Check if too large
            if img.size[0] * img.size[1] > 50_000_000:  # 50 megapixels
                return None, f"Image too large: {img.size}"

            return img.copy(), None

    except Exception as e:
        return None, f"Image processing error: {str(e)}"

# Usage
image, error = safe_image_open(image_path)
if error:
    print(f"Skipping {image_path}: {error}")
    continue
```

### Processing Issues

#### 1. Poor Classification Results

**Problem**: Images consistently misclassified

**Solutions**:

```python
# Enhance classification prompt
ENHANCED_CLASSIFICATION_PROMPT = """Carefully analyze the provided image.

Look for these indicators:
- FORM: Contains structured layouts, forms, cards with metadata, coin images with labels
- TEXT_PAGE: Primarily text content, manuscripts, printed pages, documents
- EMPTY_PAGE: Blank or nearly blank pages, separator pages

1. Classify the image content type. The type must be one of: "form", "text_page", "empty_page".
2. Look for any handwritten text (cursive, handwriting, annotations). This should be a boolean value.

Return ONLY a JSON object: {"image_type": "form|text_page|empty_page", "handwritten_content": true|false}"""

# Add confidence scoring if needed
def validate_classification_with_confidence(result):
    # Add manual validation for edge cases
    if result.get("confidence", 1.0) < 0.7:
        # Flag for manual review
        result["needs_manual_review"] = True
    return result
```

#### 2. OCR Quality Issues

**Problem**: Poor text extraction quality

**Solutions**:

```python
# Preprocess images for better OCR
def preprocess_for_ocr(image):
    # Convert to grayscale
    gray = image.convert('L')

    # Enhance contrast
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Contrast(gray)
    enhanced = enhancer.enhance(2.0)

    # Resize if too small
    if enhanced.size[0] < 1000:
        scale_factor = 1000 / enhanced.size[0]
        new_size = (int(enhanced.size[0] * scale_factor),
                   int(enhanced.size[1] * scale_factor))
        enhanced = enhanced.resize(new_size, Image.Resampling.LANCZOS)

    return enhanced

# Try different OCR configurations
def enhanced_tesseract_ocr(image_path):
    try:
        image = Image.open(image_path)
        preprocessed = preprocess_for_ocr(image)

        # Try different PSM modes
        psm_modes = [6, 3, 11, 13]  # Different page segmentation modes

        for psm in psm_modes:
            try:
                config = f'--psm {psm} -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:!?-()[]{}"\' '
                text = pytesseract.image_to_string(preprocessed, config=config)
                if len(text.strip()) > 10:  # Minimum viable text
                    return text, None
            except:
                continue

        return "", "No viable OCR result"
    except Exception as e:
        return "", str(e)
```

#### 3. Coin Extraction Failures

**Problem**: Coins not detected or poorly cropped

**Solutions**:

```python
# Enhanced bounding box validation
def validate_bounding_box(bbox, image_size):
    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
    img_w, img_h = image_size

    # Check bounds
    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
        return False, "Bounding box out of image bounds"

    # Check minimum size
    if w < 50 or h < 50:
        return False, "Bounding box too small"

    # Check aspect ratio (coins should be roughly circular)
    aspect_ratio = w / h
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        return False, "Unrealistic aspect ratio for coin"

    return True, None

# Enhanced margin detection
def enhanced_margin_detection(original_image, bbox):
    # Try multiple margin detection strategies
    strategies = [
        {"initial": 20, "max": 80, "increment": 5},
        {"initial": 10, "max": 50, "increment": 3},
        {"initial": 30, "max": 120, "increment": 10}
    ]

    for strategy in strategies:
        margin = _find_optimal_margin(
            original_image, bbox,
            initial_margin=strategy["initial"],
            max_margin=strategy["max"],
            increment=strategy["increment"]
        )

        # Test crop quality
        cropped = crop_with_margin(original_image, bbox, margin)
        if is_good_crop_quality(cropped):
            return margin

    # Fallback to default
    return CROP_MARGIN_PIXELS

def is_good_crop_quality(cropped_image):
    # Check if crop contains sufficient non-background content
    # Implementation depends on specific requirements
    return True
```

### Performance Issues

#### 1. Slow Processing Speed

**Problem**: Processing takes too long

**Solutions**:

```python
# Profile processing time
import time
from functools import wraps

def profile_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__}: {end-start:.2f}s")
        return result
    return wrapper

# Optimize image loading
def optimize_image_loading(image_path, max_size=2048):
    with Image.open(image_path) as img:
        # Resize large images
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Convert to RGB once
        if img.mode != 'RGB':
            img = img.convert('RGB')

        return img.copy()

# Parallel processing for independent operations
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

def parallel_ocr_processing(image_paths):
    max_workers = min(multiprocessing.cpu_count(), 4)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single_ocr, image_paths))

    return results
```

#### 2. High Memory Usage

**Problem**: Memory usage grows over time

**Solutions**:

```python
# Memory cleanup utilities
import gc
import torch

def cleanup_memory():
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Process with memory management
def memory_managed_batch_processing(image_paths, batch_size=10):
    total_processed = 0

    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]

        # Process batch
        for image_path in batch:
            try:
                result = process_single_image(image_path)
                total_processed += 1
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        # Cleanup after each batch
        cleanup_memory()

        # Memory usage check
        if psutil.virtual_memory().percent > 85:
            print("Warning: High memory usage, taking break...")
            time.sleep(5)
            cleanup_memory()

    return total_processed

# Monitor memory during processing
class MemoryMonitor:
    def __init__(self, threshold=80):
        self.threshold = threshold

    def check(self):
        usage = psutil.virtual_memory().percent
        if usage > self.threshold:
            print(f"Memory usage: {usage:.1f}% - Running cleanup")
            cleanup_memory()
            return True
        return False
```

### Data Quality Issues

#### 1. Inconsistent Output Format

**Problem**: JSON structure varies between runs

**Solutions**:

```python
# Strict output validation
def validate_and_normalize_output(result):
    """Ensure consistent output format."""
    normalized = {
        "image_path_original": result.get("image_path_original", ""),
        "image_type": result.get("image_type", "unknown"),
        "handwritten_content": bool(result.get("handwritten_content", False)),
        "data": result.get("data", {}),
        "images_extracted": result.get("images_extracted", []),
        "status": result.get("status", "unknown"),
        "error_message": result.get("error_message", None)
    }

    # Validate required fields
    if normalized["image_type"] not in ["form", "text_page", "empty_page"]:
        normalized["status"] = "validation_failed"
        normalized["error_message"] = f"Invalid image_type: {normalized['image_type']}"

    return normalized

# Schema validation with detailed errors
def detailed_schema_validation(data, schema):
    try:
        validate(data, schema)
        return True, None
    except ValidationError as e:
        error_path = " -> ".join(str(x) for x in e.absolute_path)
        return False, f"Schema error at {error_path}: {e.message}"
```

#### 2. Missing or Incomplete Data

**Problem**: Some fields consistently missing

**Solutions**:

```python
# Enhanced prompts with examples
ENHANCED_FORM_PROMPT = """Analyze this image containing a coin documentation form.

REQUIRED OUTPUTS:
1. Identify ALL coins visible in the image
2. For each coin, provide:
   - Unique ID (e.g., "coin_1", "coin_2")
   - Brief description of what you see
   - Bounding box coordinates (x, y, width, height from top-left)

3. Extract ALL visible text fields from information cards/forms:
   - Atelier (workshop/mint)
   - Date (dating information)
   - Faussaire (counterfeiter information)
   - Provenance (origin/history)
   - Poids (weight)
   - Diamètre (diameter)
   - Orientation des axes (axis orientation)
   - Métal (metal composition)
   - Technique (manufacturing method)
   - Publication (publication reference)
   - Négatifs (negative numbers)
   - Remarques (additional remarks/notes)

If a field is not visible or present, use null for that field.
Ensure bounding boxes are within image boundaries and have positive dimensions.

Return ONLY a JSON object following this exact structure:
{
  "coins": [
    {
      "id": "coin_1",
      "description": "detailed description of the coin",
      "bounding_box": {"x": 100, "y": 50, "width": 150, "height": 150}
    }
  ],
  "card_fields": {
    "Atelier": "extracted text or null",
    "Date": "extracted text or null",
    "Faussaire": "extracted text or null",
    "Provenance": "extracted text or null",
    "Poids": "extracted text or null",
    "Diamètre": "extracted text or null",
    "Orientation des axes": "extracted text or null",
    "Métal": "extracted text or null",
    "Technique": "extracted text or null",
    "Publication": "extracted text or null",
    "Négatifs": "extracted text or null",
    "Remarques": "extracted text or null"
  }
}"""

# Field completion checking
def check_field_completion(result):
    """Check which fields are consistently missing."""
    if result.get("image_type") == "form":
        form_data = result.get("data", {}).get("form_data", {})
        card_fields = form_data.get("card_fields", {})

        missing_fields = []
        expected_fields = ["Atelier", "Date", "Métal", "Poids", "Diamètre"]

        for field in expected_fields:
            if not card_fields.get(field):
                missing_fields.append(field)

        if missing_fields:
            print(f"Missing fields: {missing_fields}")
            return False

    return True
```

### File System Issues

#### 1. Path and Permission Problems

**Problem**: File access errors, permission denied

**Solutions**:

```python
import os
import stat

def check_file_permissions(file_path):
    """Check file accessibility."""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return False, "File does not exist"

        # Check read permissions
        if not os.access(file_path, os.R_OK):
            return False, "No read permission"

        # Check file size
        size = os.path.getsize(file_path)
        if size == 0:
            return False, "File is empty"

        return True, None
    except Exception as e:
        return False, str(e)

def safe_create_directory(dir_path):
    """Safely create directory with error handling."""
    try:
        os.makedirs(dir_path, exist_ok=True)

        # Test write permissions
        test_file = os.path.join(dir_path, "test_write.tmp")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)

        return True, None
    except PermissionError:
        return False, "Permission denied"
    except Exception as e:
        return False, str(e)

# Windows-specific path handling
def normalize_path(path):
    """Normalize paths for cross-platform compatibility."""
    # Convert forward slashes to backslashes on Windows
    if os.name == 'nt':
        path = path.replace('/', '\\')
    else:
        path = path.replace('\\', '/')

    # Resolve relative paths
    return os.path.abspath(path)
```

#### 2. Disk Space Issues

**Problem**: Running out of disk space during processing

**Solutions**:

```python
import shutil

def check_disk_space(path, required_gb=10):
    """Check available disk space."""
    try:
        total, used, free = shutil.disk_usage(path)
        free_gb = free / (1024**3)

        if free_gb < required_gb:
            return False, f"Insufficient disk space: {free_gb:.1f}GB available, {required_gb}GB required"

        return True, f"{free_gb:.1f}GB available"
    except Exception as e:
        return False, str(e)

def cleanup_temp_files(temp_dir):
    """Clean up temporary files to free space."""
    try:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith(('.tmp', '.temp', '.cache')):
                    os.remove(os.path.join(root, file))
    except Exception as e:
        print(f"Cleanup error: {e}")
```

## Debugging Tools

### Logging Configuration

```python
import logging
from datetime import datetime

def setup_logging(log_level=logging.INFO):
    """Set up comprehensive logging."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # File handler
    log_filename = f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter(log_format))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=[file_handler, console_handler]
    )

    return logging.getLogger(__name__)

# Usage
logger = setup_logging()
logger.info("Starting processing pipeline")
```

### Debug Mode Configuration

```python
DEBUG_MODE = True  # Set to True for detailed debugging

def debug_save_intermediate_results(stage, data, image_path):
    """Save intermediate results for debugging."""
    if not DEBUG_MODE:
        return

    debug_dir = "debug_outputs"
    os.makedirs(debug_dir, exist_ok=True)

    # Save stage result
    debug_file = os.path.join(
        debug_dir,
        f"{os.path.basename(image_path)}_{stage}_debug.json"
    )

    with open(debug_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Debug data saved: {debug_file}")

# Enhanced error reporting
def detailed_error_report(error, context):
    """Generate detailed error report."""
    import traceback

    report = {
        "timestamp": datetime.now().isoformat(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context,
        "traceback": traceback.format_exc(),
        "system_info": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "available_memory": psutil.virtual_memory().available / (1024**3),
            "gpu_available": torch.cuda.is_available()
        }
    }

    # Save error report
    error_file = f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(error_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Error report saved: {error_file}")
    return report
```

## Getting Help

### Support Resources

1. **Check Documentation**:

   - [Technical Documentation](TECHNICAL_DOCUMENTATION.md)
   - [User Guide](USER_GUIDE.md)
   - [API Reference](API_REFERENCE.md)

2. **Search Known Issues**:

   - GitHub Issues: [repository issues](https://github.com/telota/imagines-nummorum-vlm-data-extraction/issues)
   - Common error patterns in this troubleshooting guide

3. **Collect Debug Information**:

   ```bash
   # System information
   python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
   pip list | grep -E "(torch|transformers|PIL)"

   # Error logs
   tail -n 50 processing_*.log
   ```

4. **Contact Support**:
   - Email: tim.westphal@bbaw.de
   - Include system information and error logs
   - Provide minimal reproducible example if possible

### Reporting Issues

When reporting issues, please include:

1. **System Information**:

   - Operating system and version
   - Python version
   - GPU information (if applicable)
   - Available RAM and disk space

2. **Error Details**:

   - Full error message and traceback
   - Steps to reproduce the issue
   - Expected vs. actual behavior

3. **Environment Details**:

   - Package versions (`pip list`)
   - Configuration settings used
   - Sample input files (if possible)

4. **Attempted Solutions**:
   - What troubleshooting steps you've already tried
   - Any partial successes or workarounds found
