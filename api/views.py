from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from rest_framework.permissions import AllowAny
import pytesseract
from PIL import Image
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer
import torch
import io
import re
import os
from django.views.generic import TemplateView
from django.http import JsonResponse
from django.conf import settings

def find_tesseract_executable():
    """Try to find Tesseract executable in common locations"""
    # First, check if it's in PATH
    import shutil
    tesseract_path = shutil.which('tesseract')
    if tesseract_path:
        return tesseract_path
    
    # Common Windows installation paths
    common_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', '')),
        r'D:\Program Files\Tesseract-OCR\tesseract.exe',
        r'D:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return None

# Configure Tesseract path (useful for Windows users)
# Priority: settings.py > environment variable > auto-detect
tesseract_found = False
tesseract_path_used = None

if hasattr(settings, 'TESSERACT_CMD'):
    tesseract_path_used = settings.TESSERACT_CMD
    pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
    tesseract_found = True
    print(f"Tesseract path set from settings.py: {settings.TESSERACT_CMD}")
elif os.getenv('TESSERACT_CMD'):
    tesseract_path_used = os.getenv('TESSERACT_CMD')
    pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_CMD')
    tesseract_found = True
    print(f"Tesseract path set from environment variable: {tesseract_path_used}")
else:
    # Try to auto-detect
    detected_path = find_tesseract_executable()
    if detected_path:
        tesseract_path_used = detected_path
        pytesseract.pytesseract.tesseract_cmd = detected_path
        tesseract_found = True
        print(f"Tesseract path auto-detected: {detected_path}")

if tesseract_path_used:
    # Verify the path exists
    if not os.path.exists(tesseract_path_used):
        print(f"WARNING: Tesseract path set but file does not exist: {tesseract_path_used}")
    else:
        print(f"Tesseract executable found at: {tesseract_path_used}")

def check_tesseract_available():
    """Check if Tesseract OCR is available"""
    current_path = getattr(pytesseract.pytesseract, 'tesseract_cmd', None)
    print(f"Checking Tesseract availability. Current path: {current_path}")
    
    try:
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract version check successful: {version}")
        return True, None
    except pytesseract.TesseractNotFoundError as e:
        print(f"TesseractNotFoundError: {e}")
        print(f"Current tesseract_cmd: {current_path}")
        
        # Check if the path exists
        if current_path and os.path.exists(current_path):
            print(f"Path exists but Tesseract still not found. This might be a permissions issue.")
        elif current_path:
            print(f"Path does not exist: {current_path}")
        
        # Try to find it again
        detected_path = find_tesseract_executable()
        if detected_path:
            print(f"Trying auto-detected path: {detected_path}")
            pytesseract.pytesseract.tesseract_cmd = detected_path
            try:
                version = pytesseract.get_tesseract_version()
                print(f"Successfully found Tesseract at: {detected_path}")
                return True, None
            except Exception as retry_e:
                print(f"Retry also failed: {retry_e}")
        
        # Build error message with current path info
        error_details = f"Current configured path: {current_path if current_path else 'None'}\n"
        if current_path:
            error_details += f"Path exists: {os.path.exists(current_path) if current_path else 'N/A'}\n"
        
        install_instructions = (
            "Tesseract OCR is not installed or not found.\n\n"
            f"{error_details}\n"
            "To install Tesseract OCR on Windows:\n"
            "1. Download the installer from: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "2. Run the installer and install to the default location (C:\\Program Files\\Tesseract-OCR)\n"
            "3. OR add Tesseract to your system PATH\n"
            "4. OR set the path in settings.py: TESSERACT_CMD = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'\n"
            "5. Restart your Django server after installation"
        )
        return False, install_instructions
    except Exception as e:
        error_msg = f"Error checking Tesseract: {str(e)}\nCurrent path: {current_path}"
        print(error_msg)
        return False, error_msg


# Global model cache to avoid reloading models on every request
_model_cache = {
    'translation_model': None,
    'translation_tokenizer': None,
}

def get_translation_model():
    """Get or load the translation model (cached)"""
    if _model_cache['translation_model'] is None:
        try:
            mt_model_name = "Helsinki-NLP/opus-mt-en-fa"
            _model_cache['translation_tokenizer'] = MarianTokenizer.from_pretrained(mt_model_name)
            _model_cache['translation_model'] = MarianMTModel.from_pretrained(mt_model_name)
        except Exception as e:
            print(f"Error loading translation model: {e}")
    return _model_cache['translation_model'], _model_cache['translation_tokenizer']

def generate_simple_explanation(extracted_text):
    """
    Generate a simple, user-friendly explanation of the prescription in English.
    This function parses the prescription text and creates an easy-to-understand explanation.
    """
    if not extracted_text or not extracted_text.strip():
        return "No text could be extracted from the prescription image. Please ensure the image is clear and readable."
    
    # Clean the extracted text
    text = extracted_text.strip()
    
    # Common prescription patterns to extract
    medications = []
    dosages = []
    instructions = []
    
    # Look for medication names (common patterns)
    # This is a simplified parser - in production, you'd want a more sophisticated NLP approach
    lines = text.split('\n')
    
    explanation_parts = []
    explanation_parts.append("Here's what your prescription says in simple terms:\n\n")
    
    # Extract medications and dosages
    medication_keywords = ['mg', 'ml', 'tablet', 'capsule', 'drops', 'cream', 'ointment', 'gel']
    found_medications = False
    
    for line in lines:
        line_lower = line.lower().strip()
        if any(keyword in line_lower for keyword in medication_keywords):
            found_medications = True
            # Clean up the line
            clean_line = ' '.join(line.split())
            if clean_line:
                explanation_parts.append(f"• {clean_line}\n")
    
    if not found_medications:
        # If no clear medication pattern found, provide a general explanation
        explanation_parts.append("The prescription contains the following information:\n")
        # Show first few lines of extracted text
        important_lines = [line.strip() for line in lines[:5] if line.strip()]
        for line in important_lines:
            explanation_parts.append(f"• {line}\n")
    
    explanation_parts.append("\n")
    explanation_parts.append("Important Notes:\n")
    explanation_parts.append("• Always follow your doctor's instructions exactly\n")
    explanation_parts.append("• Take medications at the prescribed times\n")
    explanation_parts.append("• Complete the full course of medication if prescribed\n")
    explanation_parts.append("• Contact your doctor if you have any questions or concerns\n")
    
    explanation = ''.join(explanation_parts)
    
    # If the text is very short, provide a more detailed explanation
    if len(text) < 50:
        explanation = f"The prescription image shows: {text}\n\n"
        explanation += "Please consult with your pharmacist or doctor for detailed instructions on how to take this medication."
    
    return explanation

def translate_to_persian(text):
    """Translate English text to Persian (Farsi)"""
    try:
        model, tokenizer = get_translation_model()
        if model is None or tokenizer is None:
            return "مدل ترجمه در دسترس نیست. لطفاً تنظیمات سرور را بررسی کنید."
        
        # Split long text into chunks if needed (MarianMT has token limits)
        max_length = 512
        if len(text) > max_length:
            # Split into sentences and translate in chunks
            sentences = re.split(r'[.!?]\s+', text)
            translated_parts = []
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < max_length:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        inputs = tokenizer(current_chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
                        translated = model.generate(**inputs)
                        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
                        translated_parts.append(translated_text)
                    current_chunk = sentence + ". "
            
            # Translate remaining chunk
            if current_chunk:
                inputs = tokenizer(current_chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
                translated = model.generate(**inputs)
                translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
                translated_parts.append(translated_text)
            
            return " ".join(translated_parts)
        else:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            translated = model.generate(**inputs)
            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
            return translated_text
    except Exception as e:
        # Return a fallback message in Persian
        print(f"Translation error: {str(e)}")
        return f"خطا در ترجمه: {str(e)}. توضیحات انگلیسی در بالا در دسترس است."

# --- API Endpoint for Prescription Upload ---
class UploadPrescriptionView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    permission_classes = [AllowAny]  # Allow unauthenticated access

    def post(self, request, format=None):
        try:
            image_file = request.FILES.get('image')
            if not image_file:
                return JsonResponse({'error': 'No image uploaded.'}, status=400)

            # Check if Tesseract is available before processing
            tesseract_available, tesseract_error = check_tesseract_available()
            if not tesseract_available:
                error_message = (
                    "Tesseract OCR is not installed or not found.\n\n"
                    "QUICK FIX:\n"
                    "1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki\n"
                    "2. Install it (choose 'Add to PATH' during installation)\n"
                    "3. Restart your Django server\n\n"
                    "OR manually set the path in Theranous/settings.py:\n"
                    "TESSERACT_CMD = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'\n\n"
                    "See TESSERACT_INSTALLATION.md for detailed instructions."
                )
                return JsonResponse({
                    'error': error_message
                }, status=500)

            # Extract text from image using OCR
            extracted_text = ""
            try:
                image = Image.open(image_file)
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                extracted_text = pytesseract.image_to_string(image, lang='eng')
            except pytesseract.TesseractNotFoundError:
                return JsonResponse({
                    'error': 'Tesseract OCR is not installed or not found in PATH. Please install Tesseract OCR on your system. For Windows, download from: https://github.com/UB-Mannheim/tesseract/wiki'
                }, status=500)
            except Exception as e:
                error_msg = str(e)
                print(f"OCR Error: {error_msg}")
                return JsonResponse({
                    'error': f'Error processing image with OCR: {error_msg}. Please ensure the image is clear and Tesseract is properly installed.'
                }, status=400)

            # Generate simple explanation in English
            try:
                explanation_en = generate_simple_explanation(extracted_text)
            except Exception as e:
                print(f"Explanation generation error: {str(e)}")
                explanation_en = f"Explanation generation error: {str(e)}. Extracted text: {extracted_text[:200]}"
            
            # Translate to Persian (make it optional - don't fail if translation fails)
            explanation_fa = ""
            try:
                explanation_fa = translate_to_persian(explanation_en)
            except Exception as e:
                print(f"Translation error: {str(e)}")
                explanation_fa = "خطا در ترجمه. لطفاً توضیحات انگلیسی را مشاهده کنید."

            return JsonResponse({
                'extracted_text': extracted_text,
                'explanation_en': explanation_en,
                'explanation_fa': explanation_fa
            })
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Unexpected error: {error_trace}")
            return JsonResponse({
                'error': f'An unexpected error occurred: {str(e)}. Please check the server logs for more details.'
            }, status=500)

# --- Frontend View for Upload Form ---
class PrescriptionFormView(TemplateView):
    template_name = "prescription_form.html"

    def get(self, request, *args, **kwargs):
        # Render the upload form HTML
        return render(request, self.template_name)
