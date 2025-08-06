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
        
        # Comprehensive list of common medicine names
        self.known_medicines = [
            # Pain relievers and anti-inflammatory
            'aspirin', 'ibuprofen', 'acetaminophen', 'paracetamol', 'naproxen', 'diclofenac',
            'celecoxib', 'meloxicam', 'indomethacin', 'ketorolac',
            
            # Antibiotics
            'amoxicillin', 'penicillin', 'erythromycin', 'azithromycin', 'clarithromycin',
            'ciprofloxacin', 'levofloxacin', 'doxycycline', 'cephalexin', 'clindamycin',
            'metronidazole', 'trimethoprim', 'sulfamethoxazole',
            
            # Cardiovascular medications
            'lisinopril', 'enalapril', 'losartan', 'valsartan', 'amlodipine', 'nifedipine',
            'metoprolol', 'atenolol', 'carvedilol', 'propranolol', 'diltiazem', 'verapamil',
            'atorvastatin', 'simvastatin', 'rosuvastatin', 'pravastatin', 'lovastatin',
            'warfarin', 'heparin', 'clopidogrel', 'aspirin',
            
            # Diabetes medications
            'metformin', 'glipizide', 'glyburide', 'insulin', 'glimepiride', 'pioglitazone',
            'sitagliptin', 'metformin', 'rosiglitazone',
            
            # Gastrointestinal medications
            'omeprazole', 'lansoprazole', 'pantoprazole', 'esomeprazole', 'ranitidine',
            'famotidine', 'cimetidine', 'sucralfate', 'metoclopramide',
            
            # Respiratory medications
            'albuterol', 'salbutamol', 'fluticasone', 'budesonide', 'montelukast',
            'theophylline', 'ipratropium', 'tiotropium',
            
            # Corticosteroids
            'prednisone', 'prednisolone', 'hydrocortisone', 'dexamethasone', 'methylprednisolone',
            'betamethasone', 'triamcinolone',
            
            # Diuretics
            'furosemide', 'hydrochlorothiazide', 'spironolactone', 'amiloride', 'triamterene',
            'chlorthalidone', 'indapamide',
            
            # Mental health medications
            'sertraline', 'fluoxetine', 'paroxetine', 'citalopram', 'escitalopram',
            'venlafaxine', 'duloxetine', 'bupropion', 'trazodone', 'mirtazapine',
            'lorazepam', 'alprazolam', 'clonazepam', 'diazepam',
            
            # Thyroid medications
            'levothyroxine', 'liothyronine', 'methimazole', 'propylthiouracil',
            
            # Common supplements and vitamins
            'vitamin d', 'vitamin b12', 'folic acid', 'iron', 'calcium', 'magnesium'
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
        """Generate comprehensive explanation for a medicine using Flan-T5"""
        if not self.model or not self.tokenizer:
            return self._get_fallback_explanation(medicine_name)
        
        try:
            # Create an optimized prompt for better medicine explanations
            prompt = f"""Provide a clear, comprehensive explanation about the medication {medicine_name}. 
            Include: 1) What it is used for (main purpose), 2) How it works in the body, 3) Common conditions it treats, 
            4) Important usage information. Write in simple, easy-to-understand language for patients."""
            
            # Tokenize input with optimized parameters
            inputs = self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)
            
            # Generate explanation with optimized parameters for better quality
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=300,  # Increased for more comprehensive explanations
                    min_length=80,   # Increased minimum for more detailed info
                    num_beams=6,     # More beams for better quality
                    temperature=0.3, # Lower temperature for more focused, accurate responses
                    do_sample=True,
                    repetition_penalty=1.2,  # Reduce repetition
                    length_penalty=1.0,      # Encourage appropriate length
                    pad_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )
            
            # Decode the generated text
            explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up and format the explanation
            explanation = self._clean_and_format_explanation(explanation, medicine_name)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation for {medicine_name}: {str(e)}")
            return self._get_fallback_explanation(medicine_name)
    
    def _clean_and_format_explanation(self, explanation: str, medicine_name: str) -> str:
        """Clean up and format the generated explanation for better readability"""
        if not explanation:
            return self._get_fallback_explanation(medicine_name)
        
        # Remove the original prompt if it appears in the response
        explanation = explanation.replace("Provide a clear, comprehensive explanation about the medication", "")
        explanation = explanation.replace("Include: 1) What it is used for", "")
        
        # Clean up common artifacts
        explanation = explanation.strip()
        
        # Remove repetitive content
        sentences = explanation.split('.')
        unique_sentences = []
        seen_content = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Ignore very short fragments
                # Check for substantial similarity to avoid repetition
                sentence_lower = sentence.lower()
                is_duplicate = False
                for seen in seen_content:
                    if len(set(sentence_lower.split()) & set(seen.split())) / max(len(sentence_lower.split()), 1) > 0.7:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_sentences.append(sentence)
                    seen_content.add(sentence_lower)
        
        # Limit to most relevant sentences (max 4-5 for readability)
        unique_sentences = unique_sentences[:5]
        
        if unique_sentences:
            formatted_explanation = '. '.join(unique_sentences)
            if not formatted_explanation.endswith('.'):
                formatted_explanation += '.'
            
            # Ensure the medicine name is mentioned in the explanation
            if medicine_name.lower() not in formatted_explanation.lower():
                formatted_explanation = f"{medicine_name} is a medication. {formatted_explanation}"
            
            return formatted_explanation
        
        return self._get_fallback_explanation(medicine_name)
    
    def _get_fallback_explanation(self, medicine_name: str) -> str:
        """Provide comprehensive fallback explanations when model is not available"""
        fallback_explanations = {
            'aspirin': 'Aspirin is a widely used medication that belongs to a group called nonsteroidal anti-inflammatory drugs (NSAIDs). It works by blocking certain chemicals in your body that cause pain, inflammation, and fever. Aspirin is commonly used to treat headaches, muscle aches, toothaches, and reduce fever. In low doses, it is also prescribed to help prevent heart attacks and strokes by preventing blood clots. Always follow your doctor\'s instructions when taking aspirin.',
            
            'ibuprofen': 'Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) that reduces pain, inflammation, and fever. It works by blocking enzymes that produce prostaglandins, which are chemicals that cause inflammation and pain. Ibuprofen is effective for treating headaches, dental pain, menstrual cramps, muscle aches, arthritis, and minor injuries. It typically starts working within 30-60 minutes and effects last 4-6 hours.',
            
            'acetaminophen': 'Acetaminophen (also known as paracetamol) is a pain reliever and fever reducer. Unlike NSAIDs, it works primarily in the brain to block pain signals and regulate body temperature. It is commonly used for headaches, muscle aches, arthritis, backaches, toothaches, colds, and fevers. Acetaminophen is generally gentler on the stomach than other pain relievers, making it suitable for people who cannot take NSAIDs.',
            
            'amoxicillin': 'Amoxicillin is a penicillin-type antibiotic that fights bacterial infections. It works by interfering with the bacteria\'s ability to build their cell walls, causing them to die. It is commonly prescribed for ear infections, strep throat, pneumonia, urinary tract infections, and skin infections. Amoxicillin is only effective against bacterial infections, not viral infections like colds or flu. It\'s important to complete the full course even if you feel better.',
            
            'metformin': 'Metformin is the most commonly prescribed medication for type 2 diabetes. It works by reducing the amount of glucose (sugar) produced by the liver and improving your body\'s sensitivity to insulin. This helps lower blood sugar levels and improve glucose control. Metformin may also help with weight management and is sometimes used to treat polycystic ovary syndrome (PCOS). It is usually taken with meals to reduce stomach upset.',
            
            'omeprazole': 'Omeprazole is a proton pump inhibitor (PPI) that reduces the amount of acid produced in your stomach. It works by blocking the enzyme responsible for acid production, providing relief from acid-related conditions. It is commonly used to treat gastroesophageal reflux disease (GERD), stomach ulcers, and heartburn. Omeprazole helps heal acid-damaged tissue and prevents further damage by maintaining lower stomach acid levels.',
            
            'lisinopril': 'Lisinopril is an ACE inhibitor used to treat high blood pressure (hypertension) and heart failure. It works by relaxing blood vessels, which allows blood to flow more easily and reduces the workload on the heart. By lowering blood pressure, lisinopril helps prevent strokes, heart attacks, and kidney problems. It may take several weeks to see the full benefits of this medication.',
            
            'atorvastatin': 'Atorvastatin is a statin medication used to lower cholesterol levels in the blood. It works by blocking an enzyme that your body needs to make cholesterol, thereby reducing the amount of cholesterol produced by the liver. Lower cholesterol levels help prevent heart disease, stroke, and other cardiovascular problems. Atorvastatin is most effective when combined with a healthy diet and regular exercise.',
            
            'prednisone': 'Prednisone is a corticosteroid medication that mimics cortisol, a hormone naturally produced by the adrenal glands. It has powerful anti-inflammatory and immune-suppressing effects. Prednisone is used to treat various conditions including allergic reactions, autoimmune diseases, arthritis, asthma, and certain skin conditions. It works by reducing inflammation and suppressing the immune system\'s overactive response.',
        }
        
        medicine_lower = medicine_name.lower()
        
        # Check for exact matches first
        if medicine_lower in fallback_explanations:
            return fallback_explanations[medicine_lower]
        
        # Check for partial matches (e.g., "aspirin 325mg" matches "aspirin")
        for known_medicine, explanation in fallback_explanations.items():
            if known_medicine in medicine_lower or medicine_lower in known_medicine:
                return explanation.replace(known_medicine.title(), medicine_name.title())
        
        # Generic explanation for unknown medicines
        return f"{medicine_name} is a medication prescribed by healthcare providers. For detailed information about this specific medication, including its uses, proper dosage, potential side effects, and interactions with other drugs, please consult with your doctor, pharmacist, or refer to the medication\'s official prescribing information. Always follow your healthcare provider\'s instructions when taking any medication."


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