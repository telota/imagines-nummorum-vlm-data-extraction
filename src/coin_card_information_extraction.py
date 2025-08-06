import torch
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers.models.auto.processing_auto import AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image, UnidentifiedImageError, ImageOps, ImageDraw
import os
import json
from jsonschema import validate, ValidationError
import time
import numpy as np  # For array operations and color comparison
import pytesseract  # For Tesseract OCR
from tqdm.auto import tqdm  # For progress bars
from natsort import natsorted  # For natural sorting

# --- Configuration ---
MAX_JSON_RETRIES = 3
RETRY_DELAY_SECONDS = 1
OVERWRITE_EXISTING_OUTPUT = False  # Added flag
TESSERACT_CMD = None  # Set this to your Tesseract-OCR executable path if not in PATH
# Example for Windows: r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# Example for Linux: '/usr/bin/tesseract'

# Coin extraction parameters
CROP_MARGIN_PIXELS = (
    40  # Default margin around detected bounding boxes to prevent cutoff
)
INITIAL_CROP_MARGIN = 20  # Initial margin to try (pixels)
MAX_CROP_MARGIN = 100  # Maximum margin to try (pixels)
MARGIN_INCREMENT = 5  # Amount to increase margin in each iteration (pixels)
EDGE_CHECK_WIDTH = 5  # Width of edge band to check for uniformity (pixels)
COLOR_SIMILARITY_THRESHOLD = 30  # RGB distance threshold for similar colors
EDGE_UNIFORMITY_THRESHOLD = (
    0.85  # Percentage of edge pixels that must be uniform (0.0-1.0)
)
DEBUG_VISUALIZATION = (
    True  # Set to True to save debug visualizations of margin detection
)

# OCR Strategy for 'text_page' images:
# Options: "tesseract_hocr_only", "qwen_text_only",
#          "tesseract_then_qwen_fallback", "qwen_then_tesseract_fallback", "both"
OCR_STRATEGY_FOR_TEXT_PAGES = "both"  # Example: try both

# IMPORTANT FOR TESSERACT:
# (Same Tesseract path configuration note as before)
#    try:
#        pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract' # Common Linux path
#        pytesseract.get_tesseract_version() # Check if it works
#    except Exception: # Further attempts or warning
#        pass # Keep it minimal here, errors handled in _ocr_with_tesseract


# --- Schema for Form Data Extraction ---
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
                            "height": {"type": "integer", "minimum": 1},
                        },
                        "required": ["x", "y", "width", "height"],
                    },
                },
                "required": ["id", "description", "bounding_box"],
            },
            "default": [],
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
                "Remarques": {"type": ["string", "null"]},
            },
            "default": {},
        },
    },
    "default": {"coins": [], "card_fields": {}},
}  # Ensure all card_fields from previous version are here.

# --- Prompts ---
PROMPT_CLASSIFICATION_HANDWRITING = """Analyze the provided image.
1. Classify the image content type. The type must be one of: "form", "text_page", "empty_page".
2. Determine if the image contains any handwritten text. This should be a boolean value (true or false).
Return your response ONLY as a single JSON object with two keys: "image_type" (string) and "handwritten_content" (boolean).
Example: {"image_type": "form", "handwritten_content": true}"""

PROMPT_TEXT_EXTRACTION_QWEN = """Extract all visible text from this image.
Return your response ONLY as a single JSON object with a single key: "extracted_text" (string).
Example: {"extracted_text": "This is the content of the page..."}"""

PROMPT_FORM_EXTRACTION_TEMPLATE = f"""Analyze the provided image. It might be a form or contain structured elements like coins and an information card.
Identify each coin, provide a brief description, and its bounding box (x, y, width, height where x,y is top-left).
Extract all relevant fields from the information card or form.
Return your response ONLY as a single JSON object strictly adhering to the schema provided below.
If a field is not present, use an empty string "" or null for string fields. Ensure bounding box values are positive integers.

JSON Schema for form data:
{json.dumps(FORM_DATA_VALIDATION_SCHEMA, indent=2)}

Output only the JSON object containing the form data.
"""


def load_model_and_processor(model_name, cache_dir):
    # Set Hugging Face home directory to redirect ALL cache files (including from hf-xet)
    os.environ["HF_HOME"] = os.path.abspath(cache_dir)
    print(f"Set Hugging Face cache directory to: {os.environ['HF_HOME']}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        torch_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
    else:
        torch_dtype = torch.float32
    print(
        f"Using {'bfloat16' if torch_dtype == torch.bfloat16 else 'float16' if device == 'cuda' else 'float32'} precision."
    )
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            model_name, cache_dir=cache_dir, trust_remote_code=True
        )
        print("Model and processor loaded successfully.")
        return model, processor, device
    except Exception as e:
        print(f"Error loading model or processor: {e}")
        return None, None, None


def _validate_classification_json(data):
    if not isinstance(data, dict):
        return False, "Not a dictionary"
    if "image_type" not in data or "handwritten_content" not in data:
        return False, "Missing keys"
    if data["image_type"] not in ["form", "text_page", "empty_page"]:
        return False, "Invalid image_type"
    if not isinstance(data["handwritten_content"], bool):
        return False, "handwritten_content not boolean"
    return True, None


def _validate_qwen_text_extraction_json(data):
    if not isinstance(data, dict):
        return False, "Not a dictionary"
    if "extracted_text" not in data:
        return False, "Missing 'extracted_text' key"
    if not isinstance(data.get("extracted_text"), str):
        return False, "'extracted_text' is not a string"
    return True, None


def _call_qwen_vl_with_retry(
    image_path,
    prompt_text,
    model,
    processor,
    device,
    validation_fn=None,
    schema_to_validate_jsonschema=None,
    max_tokens=1024,
    stage_name="",
):
    if not os.path.exists(image_path):
        return None, f"Image file not found: {image_path}"
    last_error_msg = "Max retries reached for " + stage_name
    for attempt in range(MAX_JSON_RETRIES):
        # tqdm.write(f"    Attempt {attempt + 1}/{MAX_JSON_RETRIES} for Qwen-VL {stage_name} on {os.path.basename(image_path)}...") # Can be verbose
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        try:
            text_prompt_for_model = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            pil_images, _ = process_vision_info(messages)
            inputs = processor(
                text=[text_prompt_for_model],
                images=pil_images,
                padding=True,
                return_tensors="pt",
            ).to(device)
        except Exception as e:
            last_error_msg = f"Input preparation error for {stage_name}: {e}"
            if attempt == MAX_JSON_RETRIES - 1:
                return None, last_error_msg
            time.sleep(RETRY_DELAY_SECONDS)
            continue
        try:
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,  # do_sample=False
            )
            generated_ids = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()
            json_start_index = response_text.find("{")
            json_end_index = response_text.rfind("}") + 1
            if (
                json_start_index != -1
                and json_end_index != -1
                and json_start_index < json_end_index
            ):
                json_string = response_text[json_start_index:json_end_index]
                parsed_json = json.loads(json_string)
                current_error = "Validation criteria not met for " + stage_name
                if validation_fn:
                    is_valid, val_error_msg = validation_fn(parsed_json)
                    if is_valid:
                        return parsed_json, None
                    current_error = (
                        f"Custom validation failed for {stage_name}: {val_error_msg}"
                    )
                elif schema_to_validate_jsonschema:
                    validate(instance=parsed_json, schema=schema_to_validate_jsonschema)
                    return parsed_json, None
                else:
                    return parsed_json, None
                last_error_msg = current_error
            else:
                last_error_msg = (
                    "Could not find valid JSON object in response for " + stage_name
                )
        except json.JSONDecodeError as jde:
            last_error_msg = f"JSONDecodeError for {stage_name}: {jde}"
        except ValidationError as ve:
            last_error_msg = f"Schema validation error for {stage_name}: {ve.message}"
        except Exception as e:
            last_error_msg = f"Model generation/processing error for {stage_name}: {e}"

        tqdm.write(
            f"      Attempt {attempt + 1} for {stage_name} on {os.path.basename(image_path)} failed: {last_error_msg.splitlines()[0]}"
        )
        if attempt < MAX_JSON_RETRIES - 1:
            time.sleep(RETRY_DELAY_SECONDS)
    return None, last_error_msg


def _perform_qwen_text_extraction(
    image_full_path, model, processor, device, stage_name_suffix=""
):
    """Helper function to call Qwen-VL for text extraction and format the result."""
    # tqdm.write(f"      Attempting Qwen-VL text extraction for {os.path.basename(image_full_path)} {stage_name_suffix}...")
    qwen_text_json, qwen_error_msg = _call_qwen_vl_with_retry(
        image_full_path,
        PROMPT_TEXT_EXTRACTION_QWEN,
        model,
        processor,
        device,
        validation_fn=_validate_qwen_text_extraction_json,
        max_tokens=3072,
        stage_name=f"Qwen-VL Text Extraction{stage_name_suffix}",
    )
    result = {
        "source": "qwen_vl",
        "type": "plain_text",
        "content": None,
        "status": "failed",
        "error_message": qwen_error_msg,
    }
    if qwen_text_json and not qwen_error_msg:
        result.update(
            {
                "content": qwen_text_json.get("extracted_text"),
                "status": "success",
                "error_message": None,
            }
        )
    return result


def _ocr_with_tesseract(image_path):
    # tqdm.write("      Attempting OCR with Tesseract (hOCR output)...") # Can be verbose
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    try:
        # Test if Tesseract is accessible after potentially setting the command
        pytesseract.get_tesseract_version()

        hocr_bytes = pytesseract.image_to_pdf_or_hocr(
            image_path, extension="hocr", timeout=30
        )
        hocr_string = hocr_bytes.decode("utf-8")
        # tqdm.write("      Tesseract hOCR generation successful.")
        return {
            "source": "tesseract",
            "type": "hocr",
            "content": hocr_string,
            "status": "success",
            "error_message": None,
        }

    except FileNotFoundError:
        msg = "Tesseract not installed or not in PATH, or TESSERACT_CMD is incorrect."
    except UnidentifiedImageError:
        msg = "Tesseract could not open/read image."
    except pytesseract.TesseractError as te:
        msg = f"Pytesseract TesseractError: {te}"
    except Exception as e:
        msg = f"Tesseract hOCR error: {e}"

    # tqdm.write(f"      Tesseract Error for {os.path.basename(image_path)}: {msg}")
    return {
        "source": "tesseract",
        "type": "hocr",
        "content": None,
        "status": "failed",
        "error_message": msg,
    }


def process_single_image_multi_stage(
    image_full_path, output_json_dir_for_image, model, processor, device, input_base_dir
):
    final_output = {
        "image_path_original": image_full_path,
        "image_type": None,
        "handwritten_content": None,
        "data": {"ocr_results": [], "form_data": None},
        "images_extracted": [],
        "status": "pending",
        "error_message": None,
    }

    # Stage 1: Classification
    # tqdm.write(f"    Stage 1: Classification for {os.path.basename(image_full_path)}...") # Can be verbose
    class_hw_data, error_msg_s1 = _call_qwen_vl_with_retry(
        image_full_path,
        PROMPT_CLASSIFICATION_HANDWRITING,
        model,
        processor,
        device,
        validation_fn=_validate_classification_json,
        max_tokens=256,
        stage_name="Classification",
    )
    if error_msg_s1 or not class_hw_data:
        final_output["status"] = "classification_failed"
        final_output["error_message"] = (
            error_msg_s1 or "Classification empty post-retries."
        )
        return final_output
    final_output["image_type"] = class_hw_data["image_type"]
    final_output["handwritten_content"] = class_hw_data["handwritten_content"]
    # tqdm.write(f"    Stage 1 Result: Type='{final_output['image_type']}', Handwritten='{final_output['handwritten_content']}'")

    # Stage 2: Content Extraction
    image_type = final_output["image_type"]
    if image_type == "empty_page":
        final_output["status"] = "success"
        final_output["data"] = {
            "ocr_results": [],
            "form_data": None,
        }  # Consistent data structure

    elif image_type == "text_page":
        # tqdm.write(f"    Stage 2: Processing text_page {os.path.basename(image_full_path)} with strategy: {OCR_STRATEGY_FOR_TEXT_PAGES}")
        final_output["data"] = {
            "ocr_results": [],
            "form_data": None,
        }  # Consistent data structure
        ran_tesseract = False
        ran_qwen_text = False
        tess_result = None
        qwen_result = None

        # Handle Tesseract first strategies
        if OCR_STRATEGY_FOR_TEXT_PAGES in [
            "tesseract_hocr_only",
            "tesseract_then_qwen_fallback",
            "both",
        ]:
            tess_result = _ocr_with_tesseract(image_full_path)
            final_output["data"]["ocr_results"].append(tess_result)
            ran_tesseract = True

        # Handle Qwen first strategies
        if OCR_STRATEGY_FOR_TEXT_PAGES in [
            "qwen_text_only",
            "qwen_then_tesseract_fallback",
            "both",
        ]:
            qwen_result = _perform_qwen_text_extraction(
                image_full_path, model, processor, device
            )
            final_output["data"]["ocr_results"].append(qwen_result)
            ran_qwen_text = True

        # Handle fallback scenarios
        if (
            OCR_STRATEGY_FOR_TEXT_PAGES == "tesseract_then_qwen_fallback"
            and ran_tesseract
            and tess_result is not None
            and tess_result["status"] == "failed"
        ):
            if not ran_qwen_text:
                tqdm.write(
                    f"      Tesseract failed, trying Qwen-VL fallback for {os.path.basename(image_full_path)}..."
                )
                qwen_result = _perform_qwen_text_extraction(
                    image_full_path,
                    model,
                    processor,
                    device,
                    stage_name_suffix=" (Fallback)",
                )
                final_output["data"]["ocr_results"].append(qwen_result)
        elif (
            OCR_STRATEGY_FOR_TEXT_PAGES == "qwen_then_tesseract_fallback"
            and ran_qwen_text
            and qwen_result is not None
            and qwen_result["status"] == "failed"
        ):
            if not ran_tesseract:
                tqdm.write(
                    f"      Qwen-VL failed, trying Tesseract fallback for {os.path.basename(image_full_path)}..."
                )
                tess_result = _ocr_with_tesseract(image_full_path)
                final_output["data"]["ocr_results"].append(tess_result)

        # Determine overall status for text page based on strategy
        # Ensure ocr_results is not empty before checking status
        successful_ocr = any(
            res["status"] == "success"
            for res in final_output["data"]["ocr_results"]
            if final_output["data"]["ocr_results"]
        )
        if successful_ocr:
            final_output["status"] = "success"
        else:
            final_output["status"] = "text_extraction_failed_all_methods"
            final_output["error_message"] = (
                "All attempted OCR methods failed for text page."
            )

    elif image_type == "form":
        # tqdm.write(f"    Stage 2: Extracting form data for {os.path.basename(image_full_path)}...")
        final_output["data"] = {
            "ocr_results": [],
            "form_data": None,
        }  # Consistent data structure
        form_data, error_msg_s2_form = _call_qwen_vl_with_retry(
            image_full_path,
            PROMPT_FORM_EXTRACTION_TEMPLATE,
            model,
            processor,
            device,
            schema_to_validate_jsonschema=FORM_DATA_VALIDATION_SCHEMA,
            max_tokens=3072,
            stage_name="Form Extraction",
        )
        if error_msg_s2_form or not form_data:
            final_output["status"] = "form_extraction_failed"
            final_output["error_message"] = (
                error_msg_s2_form or "Form extraction data empty."
            )
        else:
            final_output["data"]["form_data"] = form_data
            final_output["status"] = "success"  # Store in form_data key
            if form_data.get("coins"):
                try:
                    with Image.open(
                        image_full_path
                    ) as original_pil_image:  # Use with statement
                        base_image_filename = os.path.splitext(
                            os.path.basename(image_full_path)
                        )[0]
                        # crops_subdir_name = f"{base_image_filename}_extracted_images"
                        # crops_output_dir = os.path.join(
                        #     output_json_dir_for_image, crops_subdir_name
                        # )
                        # Get relative path from input directory to preserve structure
                        relative_path_from_input = os.path.relpath(
                            os.path.dirname(image_full_path), INPUT_IMAGE_DIRECTORY
                        )
                        if relative_path_from_input == ".":
                            crops_output_dir = OUTPUT_IMAGES_DIRECTORY
                        else:
                            crops_output_dir = os.path.join(
                                OUTPUT_IMAGES_DIRECTORY, relative_path_from_input
                            )

                        if not os.path.exists(crops_output_dir):
                            os.makedirs(crops_output_dir)
                        # Get base name of original image for filename prefixing
                        original_image_base = os.path.splitext(
                            os.path.basename(image_full_path)
                        )[0]
                        for i, coin in enumerate(form_data["coins"]):
                            bbox = coin.get("bounding_box")
                            coin_id = coin.get("id", f"coin_{i+1}")
                            if bbox and all(
                                k in bbox for k in ["x", "y", "width", "height"]
                            ):
                                if bbox["width"] > 0 and bbox["height"] > 0:
                                    # Find optimal margin to prevent cutoff using edge analysis
                                    # tqdm.write(f"      Finding optimal margin for coin '{coin_id}'...")
                                    optimal_margin, cropped_img = _find_optimal_margin(
                                        original_image=original_pil_image,
                                        bbox=bbox,
                                        initial_margin=INITIAL_CROP_MARGIN,
                                        max_margin=MAX_CROP_MARGIN,
                                        increment=MARGIN_INCREMENT,
                                    )

                                    # tqdm.write(f"      Optimal margin for coin '{coin_id}': {optimal_margin}px")

                                    # Create filename with original image name and coin count
                                    coin_count = i + 1
                                    cropped_filename = (
                                        f"{original_image_base}_{coin_count}.png"
                                    )
                                    cropped_image_save_path = os.path.join(
                                        crops_output_dir, cropped_filename
                                    )
                                    cropped_img.save(cropped_image_save_path)
                                    # Store relative path from OUTPUT_IMAGES_DIRECTORY for consistency
                                    if relative_path_from_input == ".":
                                        relative_extracted_path = cropped_filename
                                    else:
                                        relative_extracted_path = os.path.join(
                                            relative_path_from_input, cropped_filename
                                        )
                                    final_output["images_extracted"].append(
                                        relative_extracted_path
                                    )
                                else:
                                    tqdm.write(
                                        f"      Warning: Bounding box for coin '{coin_id}' in {os.path.basename(image_full_path)} has zero/negative width/height."
                                    )
                            else:
                                tqdm.write(
                                    f"      Warning: Missing/incomplete bounding_box for a coin in {os.path.basename(image_full_path)}."
                                )
                except UnidentifiedImageError:
                    final_output["error_message"] = (
                        final_output.get("error_message", "") or ""
                    ) + "; Failed to open image for cropping (unidentified format)."
                except Exception as e_crop:
                    final_output["error_message"] = (
                        final_output.get("error_message", "") or ""
                    ) + f"; Cropping error: {e_crop}"

    else:  # Unknown image type
        final_output["status"] = "unknown_image_type_unhandled"
        final_output["error_message"] = (
            f"Unhandled image type '{image_type}' from classification."
        )

    return final_output


def batch_process_images_multi_stage(input_dir, output_dir, model_name, cache_dir):
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    model, processor, device = load_model_and_processor(model_name, cache_dir)
    if model is None or processor is None:
        print("Exiting due to model/processor loading failure.")
        return

    supported_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    all_image_paths = []
    for root, dirs, files in os.walk(input_dir):
        dirs[:] = natsorted(dirs)
        files = natsorted(files)
        for filename in files:
            if filename.lower().endswith(supported_extensions):
                all_image_paths.append(os.path.join(root, filename))
    if not all_image_paths:
        print(f"No images found in '{input_dir}'.")
        return

    print(
        f"\nFound {len(all_image_paths)} images. Starting batch processing with strategy: {OCR_STRATEGY_FOR_TEXT_PAGES}"
    )
    processed_count = 0
    successful_count = 0
    failed_count = 0

    for image_full_path in tqdm(
        all_image_paths, desc="Processing Images", unit="image"
    ):
        relative_path_from_input_dir = os.path.relpath(image_full_path, input_dir)
        base, _ = os.path.splitext(relative_path_from_input_dir)
        output_json_filename = base + ".json"
        output_json_full_path = os.path.join(output_dir, output_json_filename)
        output_json_containing_dir = os.path.dirname(output_json_full_path)
        if not os.path.exists(output_json_containing_dir):
            os.makedirs(output_json_containing_dir)

        if not OVERWRITE_EXISTING_OUTPUT and os.path.exists(
            output_json_full_path
        ):  # Modified condition
            # # tqdm.write(
            #     f"  Skipping {os.path.basename(image_full_path)}, output JSON already exists and OVERWRITE_EXISTING_OUTPUT is False."
            # )
            # Try to read status from existing JSON to accurately update counts
            try:
                with open(output_json_full_path, "r", encoding="utf-8") as f_exist:
                    existing_data = json.load(f_exist)
                if existing_data.get("status") == "success":
                    successful_count += 1
                else:
                    failed_count += (
                        1  # Or a different counter for 'skipped but previously failed'
                    )
            except Exception:
                failed_count += 1  # Assume failed if cannot read or parse
            processed_count += (
                1  # Still counts as processed in terms of script iteration
            )
            continue

        result_data = process_single_image_multi_stage(
            image_full_path,
            output_json_containing_dir,
            model,
            processor,
            device,
            input_dir,
        )
        processed_count += 1
        try:
            with open(output_json_full_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            if result_data.get("status") == "success":
                successful_count += 1
            else:
                tqdm.write(
                    f"  Processed with issues: {os.path.basename(image_full_path)}. Status: {result_data.get('status')}, Err: {str(result_data.get('error_message','N/A'))[:100]}"
                )
                failed_count += 1
        except Exception as e_save:
            tqdm.write(
                f"  Error saving JSON for {os.path.basename(image_full_path)}: {e_save}"
            )
            failed_count += 1

    print("\n--- Multi-Stage Batch Processing Summary ---")
    print(f"Total images found: {len(all_image_paths)}")
    print(f"Total images processed (attempted): {processed_count}")
    print(f"Fully successful extractions (status=='success'): {successful_count}")
    print(f"Extractions with errors/failures: {failed_count}")
    print("------------------------------------------")


def _is_color_similar(color1, color2, threshold=COLOR_SIMILARITY_THRESHOLD):
    """
    Check if two RGB colors are similar based on Euclidean distance.

    Args:
        color1: First RGB color tuple or array
        color2: Second RGB color tuple or array
        threshold: Maximum Euclidean distance to be considered similar

    Returns:
        bool: True if colors are similar, False otherwise
    """
    return np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2))) <= threshold


def _is_edge_uniform(
    image,
    edge_width=EDGE_CHECK_WIDTH,
    similarity_threshold=COLOR_SIMILARITY_THRESHOLD,
    uniformity_threshold=EDGE_UNIFORMITY_THRESHOLD,
):
    """
    Check if the edge of an image has a uniform color, indicating good margin.

    Args:
        image: PIL Image object
        edge_width: Width of the edge band to check (pixels)
        similarity_threshold: Color similarity threshold
        uniformity_threshold: Required percentage of uniform edge pixels

    Returns:
        bool: True if the edge is uniform enough, False otherwise
    """
    # Convert to numpy array for faster processing
    img_array = np.array(image)
    height, width = img_array.shape[:2]

    # Skip if image is too small
    if width <= 2 * edge_width or height <= 2 * edge_width:
        return False

    # Sample colors from the four corners (assumed to be background)
    corners = [
        img_array[0:edge_width, 0:edge_width].mean(axis=(0, 1)),  # Top-left
        img_array[0:edge_width, width - edge_width : width].mean(
            axis=(0, 1)
        ),  # Top-right
        img_array[height - edge_width : height, 0:edge_width].mean(
            axis=(0, 1)
        ),  # Bottom-left
        img_array[height - edge_width : height, width - edge_width : width].mean(
            axis=(0, 1)
        ),  # Bottom-right
    ]

    # Use the median of corner colors as the reference background color
    # This helps handle cases where one corner might be different
    background_color = np.median(corners, axis=0)

    # Check all pixels in the edge band
    edge_pixels = []

    # Top and bottom rows
    for y in range(edge_width):
        edge_pixels.extend(img_array[y, :])
        edge_pixels.extend(img_array[height - 1 - y, :])

    # Left and right columns (excluding corners already counted)
    for x in range(edge_width):
        edge_pixels.extend(img_array[edge_width : height - edge_width, x])
        edge_pixels.extend(img_array[edge_width : height - edge_width, width - 1 - x])

    # Count uniform pixels
    uniform_count = sum(
        1
        for pixel in edge_pixels
        if _is_color_similar(pixel, background_color, similarity_threshold)
    )
    total_pixels = len(edge_pixels)

    # Calculate uniformity ratio
    uniformity_ratio = uniform_count / total_pixels if total_pixels > 0 else 0

    return uniformity_ratio >= uniformity_threshold


def _find_optimal_margin(
    original_image,
    bbox,
    initial_margin=INITIAL_CROP_MARGIN,
    max_margin=MAX_CROP_MARGIN,
    increment=MARGIN_INCREMENT,
):
    """
    Find the optimal margin for coin extraction by iteratively testing margins.

    Args:
        original_image: PIL Image of the original form
        bbox: Bounding box dictionary with x, y, width, height
        initial_margin: Starting margin to try
        max_margin: Maximum margin to try
        increment: Amount to increase margin in each iteration

    Returns:
        tuple: (optimal_margin, cropped_image) with the best margin and resulting image
    """
    # Debug markers
    debug_img = None

    # Try increasing margins until we find one that works
    current_margin = initial_margin
    best_image = None
    best_margin = initial_margin

    while current_margin <= max_margin:
        # Create box with current margin
        box = (
            max(0, bbox["x"] - current_margin),
            max(0, bbox["y"] - current_margin),
            min(original_image.width, bbox["x"] + bbox["width"] + current_margin),
            min(original_image.height, bbox["y"] + bbox["height"] + current_margin),
        )

        # Skip if box dimensions are invalid
        if box[2] <= box[0] or box[3] <= box[1]:
            current_margin += increment
            continue

        # Crop the image with current margin
        cropped = original_image.crop(box)

        # Check if the edge is uniform
        if _is_edge_uniform(cropped):
            # Found a good margin, save it
            best_margin = current_margin
            best_image = cropped
            break

        # Try larger margin
        current_margin += increment

    # If no good margin found, use the maximum margin
    if best_image is None:
        box = (
            max(0, bbox["x"] - max_margin),
            max(0, bbox["y"] - max_margin),
            min(original_image.width, bbox["x"] + bbox["width"] + max_margin),
            min(original_image.height, bbox["y"] + bbox["height"] + max_margin),
        )
        best_image = original_image.crop(box)
        best_margin = max_margin

    return best_margin, best_image


if __name__ == "__main__":
    # --- Configuration ---
    INPUT_IMAGE_DIRECTORY = "../data/example_input"
    OUTPUT_JSON_DIRECTORY = "../data/example_output/extracted_data"
    OUTPUT_IMAGES_DIRECTORY = "../data/example_output/extracted_images"

    MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"
    MODEL_CACHE_DIR = "./qwen2_5_vl_model_cache"
    if not os.path.exists(MODEL_CACHE_DIR):
        os.makedirs(MODEL_CACHE_DIR)

    print("Starting multi-stage batch processing script ...")
    batch_process_images_multi_stage(
        input_dir=INPUT_IMAGE_DIRECTORY,
        output_dir=OUTPUT_JSON_DIRECTORY,
        model_name=MODEL_ID,
        cache_dir=MODEL_CACHE_DIR,
    )
