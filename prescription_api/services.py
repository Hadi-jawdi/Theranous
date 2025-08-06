import os
import re
import cv2
import numpy as np
from PIL import Image
import pytesseract
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, MarianMTModel, MarianTokenizer
from typing import List, Dict, Tuple, Optional
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

class OCRService:
    """Service for extracting text from medical prescription images"""
    
    def __init__(self):
        # Configure tesseract path if needed
        # pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
        pass
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for better OCR results"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image from {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Morphological operations to clean up the image
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
            
            return processed
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            # Fallback: return original image in grayscale
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            return img if img is not None else np.array([])
    
    def extract_text(self, image_path: str) -> str:
        """Extract text from prescription image using OCR"""
        try:
            # Preprocess the image
            processed_img = self.preprocess_image(image_path)
            
            if processed_img.size == 0:
                return ""
            
            # Configure tesseract for better medical text recognition
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,()-+/ '
            
            # Extract text
            text = pytesseract.image_to_string(
                processed_img, 
                config=custom_config,
                lang='eng'
            )
            
            # Clean up the extracted text
            text = self.clean_ocr_text(text)
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            return ""
    
    def clean_ocr_text(self, text: str) -> str:
        """Clean and normalize OCR extracted text"""
        if not text:
            return ""
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters that are likely OCR errors
        text = re.sub(r'[^\w\s.,()-+/]', '', text)
        
        return text


class MedicineDetectionService:
    """Service for detecting medicine names from OCR text"""
    
    def __init__(self):
        # Common medicine name patterns and known medications
        self.medicine_patterns = [
            r'\b\w*cillin\b',  # Antibiotics like Penicillin, Amoxicillin
            r'\b\w*mycin\b',   # Antibiotics like Erythromycin
            r'\b\w*prazole\b', # Proton pump inhibitors
            r'\b\w*olol\b',    # Beta blockers
            r'\b\w*pril\b',    # ACE inhibitors
            r'\b\w*sartan\b',  # ARB medications
            r'\b\w*statin\b',  # Cholesterol medications
            r'\b\w*pine\b',    # Calcium channel blockers
            r'\b\w*zole\b',    # Antifungals
            r'\b\w*dipine\b',  # Calcium channel blockers
        ]
        
        # Common medicine names (simplified list)
        self.known_medicines = [
            'aspirin', 'ibuprofen', 'acetaminophen', 'paracetamol',
            'amoxicillin', 'penicillin', 'erythromycin', 'azithromycin',
            'lisinopril', 'metformin', 'atorvastatin', 'simvastatin',
            'omeprazole', 'lansoprazole', 'metoprolol', 'atenolol',
            'amlodipine', 'nifedipine', 'warfarin', 'heparin',
            'insulin', 'metformin', 'glipizide', 'glyburide',
            'prednisone', 'hydrocortisone', 'dexamethasone',
            'furosemide', 'hydrochlorothiazide', 'spironolactone'
        ]
    
    def detect_medicines(self, text: str) -> List[Dict[str, any]]:
        """Detect medicine names in the given text"""
        detected_medicines = []
        text_lower = text.lower()
        
        # Check for known medicine names
        for medicine in self.known_medicines:
            if medicine.lower() in text_lower:
                # Calculate confidence based on context
                confidence = self.calculate_confidence(text_lower, medicine.lower())
                detected_medicines.append({
                    'name': medicine.title(),
                    'confidence': confidence,
                    'detection_method': 'known_list'
                })
        
        # Check for medicine patterns
        for pattern in self.medicine_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                medicine_name = match.group().strip()
                if len(medicine_name) > 3:  # Filter out very short matches
                    confidence = self.calculate_confidence(text_lower, medicine_name)
                    detected_medicines.append({
                        'name': medicine_name.title(),
                        'confidence': confidence,
                        'detection_method': 'pattern_matching'
                    })
        
        # Remove duplicates and sort by confidence
        unique_medicines = {}
        for med in detected_medicines:
            name = med['name'].lower()
            if name not in unique_medicines or med['confidence'] > unique_medicines[name]['confidence']:
                unique_medicines[name] = med
        
        result = list(unique_medicines.values())
        result.sort(key=lambda x: x['confidence'], reverse=True)
        
        return result[:10]  # Return top 10 matches
    
    def calculate_confidence(self, text: str, medicine_name: str) -> float:
        """Calculate confidence score for medicine detection"""
        base_confidence = 0.7
        
        # Boost confidence if surrounded by medical context words
        medical_context_words = [
            'tablet', 'capsule', 'mg', 'ml', 'dose', 'take', 'daily',
            'twice', 'morning', 'evening', 'prescription', 'medication'
        ]
        
        context_boost = 0
        for context_word in medical_context_words:
            if context_word in text:
                context_boost += 0.05
        
        # Reduce confidence for very short words
        if len(medicine_name) < 4:
            base_confidence -= 0.2
        
        return min(1.0, base_confidence + context_boost)


class ExplanationService:
    """Service for generating medicine explanations using Flan-T5"""
    
    def __init__(self):
        self.model_name = "google/flan-t5-base"
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
    
    def _load_model(self):
        """Load the Flan-T5 model and tokenizer"""
        try:
            logger.info("Loading Flan-T5 model...")
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Flan-T5 model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Flan-T5 model: {str(e)}")
            self.model = None
            self.tokenizer = None
    
    def generate_explanation(self, medicine_name: str) -> str:
        """Generate explanation for a medicine using Flan-T5"""
        if not self.model or not self.tokenizer:
            return self._get_fallback_explanation(medicine_name)
        
        try:
            # Create a prompt for medicine explanation
            prompt = f"Explain what {medicine_name} is used for in simple terms. Include common uses and basic information about this medication."
            
            # Tokenize input
            inputs = self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)
            
            # Generate explanation
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=200,
                    min_length=50,
                    num_beams=4,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the generated text
            explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the explanation
            explanation = self._clean_explanation(explanation)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation for {medicine_name}: {str(e)}")
            return self._get_fallback_explanation(medicine_name)
    
    def _clean_explanation(self, explanation: str) -> str:
        """Clean up the generated explanation"""
        # Remove any repetitive content
        sentences = explanation.split('.')
        unique_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in unique_sentences:
                unique_sentences.append(sentence)
        
        return '. '.join(unique_sentences[:3]) + '.' if unique_sentences else explanation
    
    def _get_fallback_explanation(self, medicine_name: str) -> str:
        """Provide fallback explanation when model is not available"""
        fallback_explanations = {
            'aspirin': 'Aspirin is a common pain reliever and anti-inflammatory medication. It is used to treat pain, fever, and inflammation. It may also be prescribed in low doses to help prevent heart attacks and strokes.',
            'ibuprofen': 'Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) used to treat pain, fever, and inflammation. It is commonly used for headaches, muscle aches, arthritis, and menstrual cramps.',
            'acetaminophen': 'Acetaminophen (also known as paracetamol) is a pain reliever and fever reducer. It is commonly used to treat mild to moderate pain and to reduce fever.',
            'amoxicillin': 'Amoxicillin is an antibiotic used to treat bacterial infections. It belongs to the penicillin family and is commonly prescribed for respiratory tract infections, ear infections, and urinary tract infections.',
            'metformin': 'Metformin is a medication used to treat type 2 diabetes. It helps control blood sugar levels by improving the way your body handles insulin.',
        }
        
        medicine_lower = medicine_name.lower()
        if medicine_lower in fallback_explanations:
            return fallback_explanations[medicine_lower]
        
        return f"{medicine_name} is a medication. Please consult with your healthcare provider or pharmacist for detailed information about this medication, including its uses, dosage, and potential side effects."


class TranslationService:
    """Service for translating explanations to Persian/Dari"""
    
    def __init__(self):
        self.model_name = "Helsinki-NLP/opus-mt-en-fa"
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
    
    def _load_model(self):
        """Load the translation model and tokenizer"""
        try:
            logger.info("Loading Persian translation model...")
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
            self.model = MarianMTModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Persian translation model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading translation model: {str(e)}")
            self.model = None
            self.tokenizer = None
    
    def translate_to_persian(self, english_text: str) -> str:
        """Translate English text to Persian/Dari"""
        if not self.model or not self.tokenizer:
            return self._get_fallback_translation(english_text)
        
        try:
            # Split long text into chunks
            chunks = self._split_text(english_text, max_length=400)
            translated_chunks = []
            
            for chunk in chunks:
                # Tokenize input
                inputs = self.tokenizer.encode(
                    chunk,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                ).to(self.device)
                
                # Generate translation
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_length=512,
                        num_beams=4,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode the translated text
                translated_chunk = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                translated_chunks.append(translated_chunk)
            
            return ' '.join(translated_chunks)
            
        except Exception as e:
            logger.error(f"Error translating text: {str(e)}")
            return self._get_fallback_translation(english_text)
    
    def _split_text(self, text: str, max_length: int = 400) -> List[str]:
        """Split text into chunks for translation"""
        sentences = text.split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk + sentence) < max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    def _get_fallback_translation(self, english_text: str) -> str:
        """Provide fallback Persian translation"""
        # Simple word-by-word translation for common medical terms
        translations = {
            'medication': 'دارو',
            'medicine': 'دارو',
            'pain': 'درد',
            'fever': 'تب',
            'infection': 'عفونت',
            'antibiotic': 'آنتی‌بیوتیک',
            'tablet': 'قرص',
            'capsule': 'کپسول',
            'doctor': 'دکتر',
            'treatment': 'درمان',
            'dosage': 'دوز',
            'side effects': 'عوارض جانبی'
        }
        
        # Simple replacement (not ideal but better than nothing)
        persian_text = english_text
        for eng, per in translations.items():
            persian_text = persian_text.replace(eng, per)
        
        return f"ترجمه فارسی: {persian_text}"


# Service instances (singletons)
ocr_service = OCRService()
medicine_detection_service = MedicineDetectionService()
explanation_service = ExplanationService()
translation_service = TranslationService()