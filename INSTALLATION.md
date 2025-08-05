# Installation Guide - Imagines Nummorum VLM Data Extraction

## System Requirements

### Hardware Requirements

#### Minimum Requirements

- **RAM**: 16GB (for 7B model)
- **Storage**: 50GB free space
- **CPU**: 4-core processor
- **GPU**: Optional but recommended

#### Recommended Requirements

- **RAM**: 64GB+ (for 32B model)
- **Storage**: 100GB+ free space
- **CPU**: 8+ core processor
- **GPU**: NVIDIA GPU with 24GB+ VRAM (RTX 4090, A100, etc.)

#### Model-Specific Requirements

| Model                   | RAM  | VRAM | Storage | Performance |
| ----------------------- | ---- | ---- | ------- | ----------- |
| Qwen2.5-VL-7B-Instruct  | 16GB | 8GB  | 30GB    | Good        |
| Qwen2.5-VL-32B-Instruct | 64GB | 24GB | 100GB   | Excellent   |

### Software Requirements

#### Operating Systems

- **Windows**: Windows 10 (version 1903+) or Windows 11
- **macOS**: macOS 10.15 (Catalina) or later
- **Linux**: Ubuntu 18.04+, CentOS 7+, or equivalent

#### Python Environment

- **Python**: 3.8+ (Python 3.9 or 3.10 recommended)
- **pip**: Latest version
- **venv**: For virtual environment management

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/telota/imagines-nummorum-vlm-data-extraction.git
cd imagines-nummorum-vlm-data-extraction
```

### 2. Create Virtual Environment

#### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Windows (Command Prompt)

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

#### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4. Install Dependencies

#### Core Dependencies

```bash
pip install -r requirements.txt
```

#### Manual Installation (if requirements.txt fails)

```bash
# Core ML libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Transformers and model utilities
pip install transformers accelerate

# Image and data processing
pip install Pillow jsonschema requests

# OCR and utilities
pip install pytesseract tqdm natsort

# Qwen-specific utilities
pip install qwen_vl_utils hf-xet
```

### 5. Install Tesseract OCR (Optional but Recommended)

#### Windows

1. Download installer from [UB Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run installer with administrator privileges
3. **Important**: Check "Add Tesseract to PATH" during installation
4. Verify installation:

```powershell
tesseract --version
```

#### macOS

```bash
# Using Homebrew
brew install tesseract

# Install additional language packs if needed
brew install tesseract-lang
```

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-eng

# Install additional languages if needed
sudo apt install tesseract-ocr-deu tesseract-ocr-fra
```

#### Verify Tesseract Installation

```bash
tesseract --version
pytesseract.get_tesseract_version()  # In Python
```

### 6. GPU Setup (Recommended)

#### NVIDIA GPU Setup

1. **Install NVIDIA drivers** (latest version)
2. **Install CUDA Toolkit** (11.8 or 12.1)
3. **Verify CUDA installation**:

```bash
nvidia-smi
nvcc --version
```

#### Verify PyTorch GPU Support

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
```

## Configuration

### 1. Create Configuration Directories

```bash
mkdir -p input_images
mkdir -p output_results
mkdir -p model_cache
```

### 2. Configure Model Cache (Optional)

Set environment variable for Hugging Face cache:

#### Windows (PowerShell)

```powershell
$env:HF_HOME = "C:\path\to\your\model_cache"
```

#### macOS/Linux

```bash
export HF_HOME="/path/to/your/model_cache"
```

### 3. Configure Tesseract Path (if not in PATH)

Edit the main script and uncomment/modify:

```python
import pytesseract
# Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Linux
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
# macOS
# pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
```

## Initial Setup and Testing

### 1. Test Basic Installation

Create `test_installation.py`:

```python
#!/usr/bin/env python3
import sys
import torch
import transformers
import PIL
from PIL import Image
import pytesseract

def test_installation():
    print("=== Installation Test ===")

    # Python version
    print(f"Python version: {sys.version}")

    # PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Transformers
    print(f"Transformers version: {transformers.__version__}")

    # PIL
    print(f"Pillow version: {PIL.__version__}")

    # Tesseract
    try:
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract version: {version}")
    except Exception as e:
        print(f"Tesseract error: {e}")

    print("=== Test completed ===")

if __name__ == "__main__":
    test_installation()
```

Run the test:

```bash
python test_installation.py
```

### 2. Download Model (First Run)

The model will be automatically downloaded on first use. For manual download:

```python
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# This will download the model to cache
model_id = "Qwen/Qwen2.5-VL-32B-Instruct"  # or 7B-Instruct
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto")
```

### 3. Test with Sample Images

1. Place test images in `input_images/` directory
2. Configure paths in the main script
3. Run a small batch test:

```bash
python src/coin_card_information_extraction.py
```

## Troubleshooting Installation

### Common Issues

#### 1. PyTorch CUDA Issues

```bash
# Uninstall and reinstall with specific CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 2. Tesseract Not Found

```python
# Add explicit path
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

#### 3. Out of Memory Errors

- Use smaller model (7B instead of 32B)
- Close other applications
- Enable virtual memory/swap
- Use CPU-only mode if necessary

#### 4. Model Download Issues

- Check internet connection
- Clear Hugging Face cache: `rm -rf ~/.cache/huggingface/`
- Use VPN if in restricted region
- Download manually and place in cache directory

#### 5. Permission Issues (Linux/macOS)

```bash
# Fix Python package permissions
sudo chown -R $USER:$USER ~/.local/
# Or use --user flag
pip install --user -r requirements.txt
```

### Performance Optimization

#### 1. Enable Mixed Precision

```python
# In model loading
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # or torch.float16
    device_map="auto"
)
```

#### 2. Optimize CUDA Settings

```python
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
```

#### 3. Batch Processing Optimization

- Process images in chunks
- Clear cache between batches
- Monitor memory usage

## Deployment Considerations

### Production Environment

1. **Containerization**: Consider Docker for consistent deployment
2. **Resource Monitoring**: Implement memory and GPU monitoring
3. **Backup Strategy**: Regular backup of model cache and outputs
4. **Security**: Secure model files and processing data
5. **Logging**: Implement comprehensive logging for troubleshooting

### Multi-GPU Setup

```python
# For multiple GPUs
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",  # Automatically distribute across GPUs
    torch_dtype=torch.bfloat16
)
```

### Environment Variables

```bash
# Set in .bashrc or equivalent
export CUDA_VISIBLE_DEVICES=0,1  # Use specific GPUs
export OMP_NUM_THREADS=8         # Control CPU threads
export HF_HOME=/path/to/cache    # Hugging Face cache
export TRANSFORMERS_CACHE=/path/to/cache  # Alternative cache setting
```

## Next Steps

After successful installation:

1. Read the [User Guide](USER_GUIDE.md) for detailed usage instructions
2. Review [Technical Documentation](TECHNICAL_DOCUMENTATION.md) for advanced features
3. Check [Troubleshooting Guide](TROUBLESHOOTING.md) for common issues
4. Explore example data in `data/example_input/` directory

## Support

For installation issues:

1. Check [Troubleshooting Guide](TROUBLESHOOTING.md)
2. Review GitHub Issues
3. Contact: tim.westphal@bbaw.de
