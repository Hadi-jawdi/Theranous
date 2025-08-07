from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
import pytesseract
from PIL import Image
from transformers import pipeline, MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import io
import re
from django.views.generic import TemplateView
from django.http import JsonResponse

# Create your views here.
class MedicineExplainer:
    """Helper class to handle medicine explanation and translation"""
    def __init__(self):
        self.explanation_tokenizer = None
        self.explanation_model = None
        self.translation_model = None
        self.translation_tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_models()

    def load_models(self):
        """Load the explanation and translation models"""
        try:
            print("🔄 Loading FLAN-T5 explanation model...")
            # Use FLAN-T5 model for zero-shot medical explanation
            model_name = "google/flan-t5-base"
            self.explanation_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.explanation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            if self.device == "cuda":
                self.explanation_model = self.explanation_model.to(self.device)
            print("✅ FLAN-T5 explanation model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading FLAN-T5 explanation model: {e}")
            self.explanation_tokenizer = None
            self.explanation_model = None

        try:
            print("🔄 Loading translation model...")
            translation_model_name = "Helsinki-NLP/opus-mt-en-fa"
            self.translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
            self.translation_model = MarianMTModel.from_pretrained(translation_model_name)
            if self.device == "cuda":
                self.translation_model = self.translation_model.to(self.device)
            print("✅ Translation model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading translation model: {e}")
            self.translation_model = None
            self.translation_tokenizer = None

    def extract_medicine_names(self, text):
        """Extract medicine names from OCR text"""
        medicines = set()
        # Pattern 1: Words ending with common suffixes (e.g., -cillin, -mycin, -zole)
        pattern1 = r'\b[A-Z][a-z]+(?:cillin|mycin|zole|pine|phen|mol|cin|fen|ide|ine|ate|ol)\b'
        matches = re.findall(pattern1, text, re.IGNORECASE)
        medicines.update([m.strip() for m in matches])

        # Pattern 2: Known common medicines
        common_meds = r'\b(?:Paracetamol|Ibuprofen|Aspirin|Amoxicillin|Metformin|Atorvastatin|Simvastatin|Omeprazole|Losartan|Amlodipine)\b'
        matches = re.findall(common_meds, text, re.IGNORECASE)
        medicines.update([m.strip() for m in matches])

        # Pattern 3: Medicine + dosage (e.g., "Amoxicillin 500mg")
        pattern3 = r'\b[A-Z][a-z]{3,}\s*\d+\s*mg\b'
        matches = re.findall(pattern3, text, re.IGNORECASE)
        for match in matches:
            clean_name = re.sub(r'\s*\d+\s*mg.*', '', match).strip()
            if clean_name:
                medicines.add(clean_name)

        # Additional: Extract capitalized words that look like medicine names
        words = re.findall(r'\b[A-Z][a-z]{4,}\b', text)
        known_medicines = {
            'paracetamol', 'acetaminophen', 'ibuprofen', 'aspirin', 'amoxicillin',
            'metformin', 'atorvastatin', 'simvastatin', 'omeprazole', 'losartan',
            'amlodipine', 'lisinopril', 'metoprolol', 'hydrochlorothiazide',
            'prednisone', 'azithromycin', 'ciprofloxacin', 'doxycycline'
        }
        exclusions = {
            'Take', 'Tablet', 'Capsule', 'Daily', 'Twice', 'Three', 'Times', 'Every',
            'Hours', 'With', 'Food', 'After', 'Before', 'Meal', 'Doctor', 'Patient',
            'Date', 'License', 'Medical', 'Center', 'Name', 'Address', 'Phone',
            'Email', 'Clinic', 'Hospital', 'Pharmacy', 'Prescription', 'Note', 'Notes',
            'Morning', 'Evening', 'Night', 'Week', 'Month', 'Year', 'Days', 'Duration'
        }

        for word in words:
            w_lower = word.lower()
            if (word not in exclusions and len(word) > 4 and
                (w_lower in known_medicines or any(w_lower.endswith(s) for s in ('in', 'ol', 'ine', 'ate', 'cillin', 'mycin')))):
                medicines.add(word)

        # Remove dosage and clean up
        filtered = []
        for med in medicines:
            clean = re.sub(r'\s*\d+\s*mg.*', '', med).strip()
            if clean and clean not in exclusions:
                filtered.append(clean)

        return list(set(filtered))

    def generate_medicine_explanation(self, medicine_name):
        """Generate explanation for a specific medicine using FLAN-T5"""
        # Try to use FLAN-T5 model if available
        if self.explanation_model and self.explanation_tokenizer:
            try:
                # Create a clear prompt for FLAN-T5
                prompt = f"Explain the medicine {medicine_name}. What is it used for, how is it taken, and what are the common side effects?"
                
                # Tokenize the prompt
                inputs = self.explanation_tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True, 
                    max_length=512
                )
                
                # Move inputs to device if using CUDA
                if self.device == "cuda":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate explanation with proper token limits
                with torch.no_grad():
                    outputs = self.explanation_model.generate(
                        **inputs,
                        max_new_tokens=200,  # Limit to ~150-200 tokens
                        num_beams=3,
                        early_stopping=True,
                        temperature=0.7,
                        do_sample=True
                    )
                
                # Decode the generated text
                explanation = self.explanation_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Format the explanation with Markdown
                if explanation and len(explanation.strip()) > 0:
                    return f"""**{medicine_name}**
{explanation.strip()}

Always follow your doctor's instructions and read the patient information leaflet."""
                    
            except Exception as e:
                print(f"Error generating FLAN-T5 explanation for {medicine_name}: {e}")
        
        # Fallback if FLAN-T5 fails or is not available
        return f"""
**{medicine_name}**
No explanation available for {medicine_name}. Please consult a doctor or pharmacist for:
- Purpose of use
- Dosage and timing
- Side effects
- Interactions with other medicines

Do not change or stop your medication without medical advice.
"""

    def translate_to_persian(self, text):
        """Translate text to Persian"""
        if not self.translation_model or not self.translation_tokenizer:
            return self._fallback_persian_translation(text)

        try:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            translated_parts = []
            for sentence in sentences:
                inputs = self.translation_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
                if self.device == "cuda":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.translation_model.generate(**inputs, max_new_tokens=150)
                translated = self.translation_tokenizer.decode(outputs[0], skip_special_tokens=True)
                translated_parts.append(translated)
            return ' '.join(translated_parts)
        except Exception as e:
            print(f"Translation error: {e}")
            return self._fallback_persian_translation(text)

    def _fallback_persian_translation(self, text):
        """Fallback Persian translation using dictionary and structured response"""
        dictionary = {
            'paracetamol': 'پارستامول',
            'acetaminophen': 'استامینوفن',
            'ibuprofen': 'ایبوپروفن',
            'amoxicillin': 'آموکسی‌سیلین',
            'aspirin': 'آسپرین',
            'pain relief': 'کاهش درد',
            'fever reduction': 'کاهش تب',
            'antibiotic': 'آنتی‌بیوتیک',
            'anti-inflammatory': 'ضد التهاب',
            'side effects': 'عوارض جانبی',
            'dosage': 'مقدار مصرف',
            'tablet': 'قرص',
            'capsule': 'کپسول',
            'daily': 'روزانه',
            'twice daily': 'دو بار در روز',
            'three times daily': 'سه بار در روز',
            'every 6 hours': 'هر ۶ ساعت',
            'with food': 'همراه با غذا',
            'after meals': 'بعد از غذا',
            'doctor': 'پزشک',
            'pharmacist': 'داروساز',
            'medicine': 'دارو',
            'medication': 'دارو',
            'prescription': 'نسخه',
            'take': 'مصرف کنید',
            'used for': 'برای استفاده از',
            'liver damage': 'آسیب کبدی',
            'stomach upset': 'ناراحتی معده',
            'allergic reactions': 'واکنش‌های آلرژیک',
            'nausea': 'تهوع',
            'diarrhea': 'اسهال',
            'bleeding risk': 'خطر خونریزی',
            'kidney issues': 'مشکلات کلیوی',
            'blood thinning': 'رقیق کردن خون',
            'bacterial infections': 'عفونت‌های باکتریایی',
            'complete the full course': 'دوره کامل را تمام کنید',
            'consult': 'مشاوره کنید',
            'healthcare provider': 'ارائه‌دهنده خدمات بهداشتی',
            'as needed': 'در صورت نیاز',
            'what it\'s used for': 'موارد استفاده',
            'how to take it': 'نحوه مصرف',
            'important notes': 'نکات مهم',
            'usually': 'معمولاً',
            'maximum': 'حداکثر',
            'per day': 'در روز',
            'may cause': 'ممکن است باعث شود',
            'generally safe': 'معمولاً ایمن',
            'when used as directed': 'هنگام استفاده طبق دستور',
            'overdose': 'مصرف بیش از حد',
            'can cause': 'می‌تواند باعث شود',
            'inflammation reduction': 'کاهش التهاب',
            'every 4-6 hours': 'هر ۴ تا ۶ ساعت',
            'every 8 hours': 'هر ۸ ساعت',
            'do not stop early': 'زودتر قطع نکنید',
            'please consult your doctor or pharmacist for personalized advice': 'لطفاً با پزشک یا داروساز خود مشورت کنید'
        }

        # Replace exact phrases first
        translated = text
        for en, fa in sorted(dictionary.items(), key=lambda x: len(x[0]), reverse=True):
            translated = re.sub(re.escape(en), fa, translated, flags=re.IGNORECASE)

        # If not enough Persian, return structured version
        persian_char_count = sum(1 for c in translated if '\u0600' <= c <= '\u06FF')
        total_chars = len(translated.replace(' ', ''))
        if total_chars > 0 and persian_char_count / total_chars < 0.3:
            structured = "توضیحات دارویی:\n"
            if 'paracetamol' in text.lower():
                structured += "**پارستامول:**\n• موارد استفاده: کاهش درد و تب\n• نحوه مصرف: ۵۰۰–۱۰۰۰ میلی‌گرم هر ۴–۶ ساعت، حداکثر ۴۰۰۰ میلی‌گرم در روز\n• نکات مهم: مصرف بیش از حد می‌تواند باعث آسیب کبدی شود.\n"
            if 'ibuprofen' in text.lower():
                structured += "**ایبوپروفن:**\n• موارد استفاده: کاهش درد و التهاب\n• نحوه مصرف: ۲۰۰–۴۰۰ میلی‌گرم هر ۴–۶ ساعت همراه با غذا\n• نکات مهم: ممکن است باعث ناراحتی معده یا مشکلات کلیوی شود.\n"
            if 'amoxicillin' in text.lower():
                structured += "**آموکسی‌سیلین:**\n• موارد استفاده: آنتی‌بیوتیک برای عفونت‌های باکتریایی\n• نحوه مصرف: ۲۵۰–۵۰۰ میلی‌گرم هر ۸ ساعت\n• نکات مهم: دوره کامل را تمام کنید. ممکن است باعث اسهال یا حساسیت شود.\n"
# ###
# from django.shortcuts import render
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework.parsers import MultiPartParser, FormParser
# from rest_framework import status
# import pytesseract
# from PIL import Image
from transformers import pipeline, MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import io
import re
from django.views.generic import TemplateView
from django.http import JsonResponse

# Create your views here.
class MedicineExplainer:
    """Helper class to handle medicine explanation and translation"""
    def __init__(self):
        self.explanation_tokenizer = None
        self.explanation_model = None
        self.translation_model = None
        self.translation_tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_models()

    def load_models(self):
        """Load the explanation and translation models"""
        try:
            print("🔄 Loading FLAN-T5 explanation model...")
            self.explanation_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            self.explanation_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
            self.explanation_model.to(self.device)
            print("✅ FLAN-T5 explanation model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading FLAN-T5 explanation model: {e}")
            self.explanation_tokenizer = None
            self.explanation_model = None

        try:
            print("🔄 Loading translation model...")
            translation_model_name = "Helsinki-NLP/opus-mt-en-fa"
            self.translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
            self.translation_model = MarianMTModel.from_pretrained(translation_model_name)
            if self.device == "cuda":
                self.translation_model = self.translation_model.to(self.device)
            print("✅ Translation model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading translation model: {e}")
            self.translation_model = None
            self.translation_tokenizer = None

    def extract_medicine_names(self, text):
        """Extract medicine names from OCR text"""
        medicines = set()
        # Pattern 1: Words ending with common suffixes (e.g., -cillin, -mycin, -zole)
        pattern1 = r'\b[A-Z][a-z]+(?:cillin|mycin|zole|pine|phen|mol|cin|fen|ide|ine|ate|ol)\b'
        matches = re.findall(pattern1, text, re.IGNORECASE)
        medicines.update([m.strip() for m in matches])

        # Pattern 2: Known common medicines
        common_meds = r'\b(?:Paracetamol|Ibuprofen|Aspirin|Amoxicillin|Metformin|Atorvastatin|Simvastatin|Omeprazole|Losartan|Amlodipine)\b'
        matches = re.findall(common_meds, text, re.IGNORECASE)
        medicines.update([m.strip() for m in matches])

        # Pattern 3: Medicine + dosage (e.g., "Amoxicillin 500mg")
        pattern3 = r'\b[A-Z][a-z]{3,}\s*\d+\s*mg\b'
        matches = re.findall(pattern3, text, re.IGNORECASE)
        for match in matches:
            clean_name = re.sub(r'\s*\d+\s*mg.*', '', match).strip()
            if clean_name:
                medicines.add(clean_name)

        # Additional: Extract capitalized words that look like medicine names
        words = re.findall(r'\b[A-Z][a-z]{4,}\b', text)
        known_medicines = {
            'paracetamol', 'acetaminophen', 'ibuprofen', 'aspirin', 'amoxicillin',
            'metformin', 'atorvastatin', 'simvastatin', 'omeprazole', 'losartan',
            'amlodipine', 'lisinopril', 'metoprolol', 'hydrochlorothiazide',
            'prednisone', 'azithromycin', 'ciprofloxacin', 'doxycycline'
        }
        exclusions = {
            'Take', 'Tablet', 'Capsule', 'Daily', 'Twice', 'Three', 'Times', 'Every',
            'Hours', 'With', 'Food', 'After', 'Before', 'Meal', 'Doctor', 'Patient',
            'Date', 'License', 'Medical', 'Center', 'Name', 'Address', 'Phone',
            'Email', 'Clinic', 'Hospital', 'Pharmacy', 'Prescription', 'Note', 'Notes',
            'Morning', 'Evening', 'Night', 'Week', 'Month', 'Year', 'Days', 'Duration'
        }

        for word in words:
            w_lower = word.lower()
            if (word not in exclusions and len(word) > 4 and
                (w_lower in known_medicines or any(w_lower.endswith(s) for s in ('in', 'ol', 'ine', 'ate', 'cillin', 'mycin')))):
                medicines.add(word)

        # Remove dosage and clean up
        filtered = []
        for med in medicines:
            clean = re.sub(r'\s*\d+\s*mg.*', '', med).strip()
            if clean and clean not in exclusions:
                filtered.append(clean)

        return list(set(filtered))

    def generate_medicine_explanation(self, medicine_name):
        """Generate explanation for a specific medicine using FLAN-T5"""
        if not self.explanation_model or not self.explanation_tokenizer:
            fallback_msg = f"No explanation available for {medicine_name}. Please consult a doctor or pharmacist."
            print(f"⚠️ Explanation model not loaded. Fallback: {fallback_msg}")
            return fallback_msg

        prompt = f"Explain the medicine {medicine_name}. What is it used for, how is it taken, and what are the common side effects?"
        print(f"📝 Prompt for FLAN-T5: {prompt}")

        inputs = self.explanation_tokenizer(prompt, return_tensors="pt").to(self.device)
        try:
            with torch.no_grad():
                outputs = self.explanation_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    eos_token_id=self.explanation_tokenizer.eos_token_id,
                    pad_token_id=self.explanation_tokenizer.pad_token_id
                )
            explanation = self.explanation_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            print(f"🧠 Explanation generated by FLAN-T5 for {medicine_name}: {explanation}")

            if not explanation or explanation.lower().startswith("explain the medicine"):
                fallback_msg = f"No explanation available for {medicine_name}. Please consult a doctor or pharmacist."
                print(f"⚠️ Empty or invalid explanation. Fallback: {fallback_msg}")
                return fallback_msg

            # Format explanation in Markdown
            formatted_explanation = f"**{medicine_name}**\n\n{explanation}"
            return formatted_explanation

        except Exception as e:
            fallback_msg = f"No explanation available for {medicine_name}. Please consult a doctor or pharmacist."
            print(f"❌ Error during explanation generation: {e}. Fallback: {fallback_msg}")
            return fallback_msg

    def translate_to_persian(self, text):
        """Translate text to Persian"""
        if not self.translation_model or not self.translation_tokenizer:
            return self._fallback_persian_translation(text)

        try:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            translated_parts = []
            for sentence in sentences:
                inputs = self.translation_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
                if self.device == "cuda":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.translation_model.generate(**inputs, max_new_tokens=150)
                translated = self.translation_tokenizer.decode(outputs[0], skip_special_tokens=True)
                translated_parts.append(translated)
            return ' '.join(translated_parts)
        except Exception as e:
            print(f"Translation error: {e}")
            return self._fallback_persian_translation(text)

    def _fallback_persian_translation(self, text):
        """Fallback Persian translation using dictionary and structured response"""
        dictionary = {
            'paracetamol': 'پارستامول',
            'acetaminophen': 'استامینوفن',
            'ibuprofen': 'ایبوپروفن',
            'amoxicillin': 'آموکسی‌سیلین',
            'aspirin': 'آسپرین',
            'pain relief': 'کاهش درد',
            'fever reduction': 'کاهش تب',
            'antibiotic': 'آنتی‌بیوتیک',
            'anti-inflammatory': 'ضد التهاب',
            'side effects': 'عوارض جانبی',
            'dosage': 'مقدار مصرف',
            'tablet': 'قرص',
            'capsule': 'کپسول',
            'daily': 'روزانه',
            'twice daily': 'دو بار در روز',
            'three times daily': 'سه بار در روز',
            'every 6 hours': 'هر ۶ ساعت',
            'with food': 'همراه با غذا',
            'after meals': 'بعد از غذا',
            'doctor': 'پزشک',
            'pharmacist': 'داروساز',
            'medicine': 'دارو',
            'medication': 'دارو',
            'prescription': 'نسخه',
            'take': 'مصرف کنید',
            'used for': 'برای استفاده از',
            'liver damage': 'آسیب کبدی',
            'stomach upset': 'ناراحتی معده',
            'allergic reactions': 'واکنش‌های آلرژیک',
            'nausea': 'تهوع',
            'diarrhea': 'اسهال',
            'bleeding risk': 'خطر خونریزی',
            'kidney issues': 'مشکلات کلیوی',
            'blood thinning': 'رقیق کردن خون',
            'bacterial infections': 'عفونت‌های باکتریایی',
            'complete the full course': 'دوره کامل را تمام کنید',
            'consult': 'مشاوره کنید',
            'healthcare provider': 'ارائه‌دهنده خدمات بهداشتی',
            'as needed': 'در صورت نیاز',
            'what it\'s used for': 'موارد استفاده',
            'how to take it': 'نحوه مصرف',
            'important notes': 'نکات مهم',
            'usually': 'معمولاً',
            'maximum': 'حداکثر',
            'per day': 'در روز',
            'may cause': 'ممکن است باعث شود',
            'generally safe': 'معمولاً ایمن',
            'when used as directed': 'هنگام استفاده طبق دستور',
            'overdose': 'مصرف بیش از حد',
            'can cause': 'می‌تواند باعث شود',
            'inflammation reduction': 'کاهش التهاب',
            'every 4-6 hours': 'هر ۴ تا ۶ ساعت',
            'every 8 hours': 'هر ۸ ساعت',
            'do not stop early': 'زودتر قطع نکنید',
            'please consult your doctor or pharmacist for personalized advice': 'لطفاً با پزشک یا داروساز خود مشورت کنید'
        }

        # Replace exact phrases first
        translated = text
        for en, fa in sorted(dictionary.items(), key=lambda x: len(x[0]), reverse=True):
            translated = re.sub(re.escape(en), fa, translated, flags=re.IGNORECASE)

        # If not enough Persian, return structured version
        persian_char_count = sum(1 for c in translated if '\u0600' <= c <= '\u06FF')
        total_chars = len(translated.replace(' ', ''))
        if total_chars > 0 and persian_char_count / total_chars < 0.3:
            structured = "توضیحات دارویی:\n"
            if 'paracetamol' in text.lower():
                structured += "**پارستامول:**\n• موارد استفاده: کاهش درد و تب\n• نحوه مصرف: ۵۰۰–۱۰۰۰ میلی‌گرم هر ۴–۶ ساعت، حداکثر ۴۰۰۰ میلی‌گرم در روز\n• نکات مهم: مصرف بیش از حد می‌تواند باعث آسیب کبدی شود.\n"
            if 'ibuprofen' in text.lower():
                structured += "**ایبوپروفن:**\n• موارد استفاده: کاهش درد و التهاب\n• نحوه مصرف: ۲۰۰–۴۰۰ میلی‌گرم هر ۴–۶ ساعت همراه با غذا\n• نکات مهم: ممکن است باعث ناراحتی معده یا مشکلات کلیوی شود.\n"
            if 'amoxicillin' in text.lower():
                structured += "**آموکسی‌سیلین:**\n• موارد استفاده: آنتی‌بیوتیک برای عفونت‌های باکتریایی\n• نحوه مصرف: ۲۵۰–۵۰۰ میلی‌گرم هر ۸ ساعت\n• نکات مهم: دوره کامل را تمام کنید. ممکن است باعث اسهال یا حساسیت شود.\n"
            if 'aspirin' in text.lower():
                structured += "**آسپرین:**\n• موارد استفاده: کاهش درد و رقیق کردن خون\n• نحوه مصرف: ۷۵–۱۰۰ میلی‌گرم روزانه برای قلب، ۳۰۰–۶۰۰ میلی‌گرم برای درد\n• نکات مهم: خطر خونریزی دارد. در کودکان استفاده نشود.\n"
            structured += "\n**نکته مهم:** با پزشک یا داروساز خود مشورت کنید."
            return structured

        return translated


# Global instance
medicine_explainer = MedicineExplainer()


class PrescriptionReaderView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        try:
            # Get uploaded image
            if 'image' not in request.FILES:
                return Response(
                    {'error': 'No image file provided'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            image_file = request.FILES['image']

            # Open and process image
            try:
                image = Image.open(image_file).convert("RGB")
            except Exception as e:
                return Response(
                    {'error': f'Invalid image file: {str(e)}'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Extract text using OCR
            try:
                extracted_text = pytesseract.image_to_string(image)
                print(f"🖼️ OCR extracted text:\n{extracted_text}")
            except Exception as e:
                return Response(
                    {'error': f'OCR failed: {str(e)}'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            if not extracted_text.strip():
                return Response(
                    {'error': 'No readable text found in the image. Please upload a clearer image.'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Extract medicine names
            medicines = medicine_explainer.extract_medicine_names(extracted_text)
            print(f"💊 Extracted medicine names: {medicines}")

            # Generate explanations for each medicine
            explanations_en = []
            explanations_fa = []
            for medicine in medicines[:5]:  # Limit to 5 for performance
                explanation_en = medicine_explainer.generate_medicine_explanation(medicine)
                explanation_fa = medicine_explainer.translate_to_persian(explanation_en)
                explanations_en.append(f"**{medicine}:**\n{explanation_en}")
                explanations_fa.append(f"**{medicine}:**\n{explanation_fa}")

            # Combine explanations
            full_explanation_en = "\n\n".join(explanations_en)
            full_explanation_fa = "\n\n".join(explanations_fa)

            if not explanations_en:
                full_explanation_en = """
No medicines were clearly identified.
**General Advice:**
- Take medications exactly as prescribed
- Read the leaflet inside the package
- Ask your pharmacist any questions
- Do not stop medication without consulting your doctor
- Report side effects immediately
"""

                full_explanation_fa = "توضیحات دارویی موجود نیست. لطفاً با پزشک یا داروساز خود مشورت کنید."

            # Return structured response
            response_data = {
                'extracted_text': extracted_text.strip(),
                'medicines_found': medicines,
                'explanation_en': full_explanation_en.strip(),
                'explanation_fa': full_explanation_fa.strip(),
                'status': 'success'
            }
            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {'error': f'An error occurred: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class PrescriptionFormView(TemplateView):
    template_name = 'prescription_form.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'Prescription Reader'














# from django.shortcuts import render
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework.parsers import MultiPartParser, FormParser
# from rest_framework import status
# import pytesseract
# from PIL import Image
# from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, MarianMTModel, MarianTokenizer
# import torch
# import io
# import re
# from django.views.generic import TemplateView
# from django.http import JsonResponse

# # Create your views here.

# class MedicineExplainer:
#     """Helper class to handle medicine explanation and translation"""
    
#     def __init__(self):
#         self.explanation_model = None
#         self.translation_model = None
#         self.translation_tokenizer = None
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.load_models()
    
#     def load_models(self):
#         """Load the explanation and translation models"""
#         try:
#             print("🔄 Loading explanation model...")
#             # Use a more stable text generation model
#             model_name = "microsoft/DialoGPT-medium"
#             self.explanation_model = pipeline(
#                 "text-generation",
#                 model=model_name,
#                 device=0 if self.device == "cuda" else -1,
#                 max_length=512,
#                 do_sample=True,
#                 temperature=0.7,
#                 pad_token_id=50256
#             )
#             print("✅ Explanation model loaded successfully")
            
#         except Exception as e:
#             print(f"❌ Error loading explanation model: {e}")
#             self.explanation_model = None
        
#         try:
#             print("🔄 Loading translation model...")
#             # Try to load a Persian translation model - use a more reliable alternative
#             translation_model_name = "Helsinki-NLP/opus-mt-en-fa"  # English to Persian
#             try:
#                 self.translation_model = MarianMTModel.from_pretrained(translation_model_name)
#                 self.translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
#                 if self.device == "cuda":
#                     self.translation_model = self.translation_model.cuda()
#                 print("✅ Translation model loaded successfully")
#             except Exception as inner_e:
#                 print(f"❌ Primary translation model failed: {inner_e}")
#                 # Fallback: try a different approach or just use Google Translate-like service
#                 print("📝 Using fallback translation approach...")
#                 self.translation_model = None
#                 self.translation_tokenizer = None
            
#         except Exception as e:
#             print(f"❌ Error loading translation model: {e}")
#             self.translation_model = None
#             self.translation_tokenizer = None
    
#     def extract_medicine_names(self, text):
#         """Extract medicine names from OCR text"""
#         # Common medicine name patterns
#         medicine_patterns = [
#             r'\b[A-Z][a-z]+(?:cillin|mycin|zole|pine|phen|mol|cin|fen|ide|ine|ate|ol)\b',  # Common suffixes
#             r'\b(?:Paracetamol|Ibuprofen|Aspirin|Amoxicillin|Metformin|Atorvastatin|Simvastatin|Omeprazole|Losartan|Amlodipine)\b',  # Common medicines
#             r'\b[A-Z][a-z]{3,}\s*\d+\s*mg\b',  # Medicine with dosage
#         ]
        
#         medicines = set()
        
#         # Extract using patterns
#         for pattern in medicine_patterns:
#             matches = re.findall(pattern, text, re.IGNORECASE)
#             medicines.update([match.strip() for match in matches])
        
#         # Extract words that look like medicine names (capitalized, not common words)
#         words = re.findall(r'\b[A-Z][a-z]{3,}\b', text)
        
#         # Exclude common non-medicine words
#         exclusions = {
#             'Take', 'Tablet', 'Capsule', 'Daily', 'Twice', 'Three', 'Times', 'Every', 'Hours', 
#             'With', 'Food', 'After', 'Before', 'Meal', 'Doctor', 'Patient', 'Date', 'License',
#             'Medical', 'Center', 'Smith', 'John', 'Sarah', 'Duration', 'Maximum', 'Minimum',
#             'Morning', 'Evening', 'Night', 'Week', 'Month', 'Year', 'Days', 'Name', 'Address',
#             'Phone', 'Email', 'Clinic', 'Hospital', 'Pharmacy', 'Prescription', 'Note', 'Notes'
#         }
        
#         # Known medicine names for validation
#         known_medicines = {
#             'paracetamol', 'acetaminophen', 'ibuprofen', 'aspirin', 'amoxicillin', 
#             'metformin', 'atorvastatin', 'simvastatin', 'omeprazole', 'losartan',
#             'amlodipine', 'lisinopril', 'metoprolol', 'hydrochlorothiazide',
#             'prednisone', 'azithromycin', 'ciprofloxacin', 'doxycycline'
#         }
        
#         for word in words:
#             if (word not in exclusions and 
#                 len(word) > 4 and 
#                 (word.lower() in known_medicines or word.lower().endswith(('in', 'ol', 'ine', 'ate')))):
#                 medicines.add(word)
        
#         # Filter and return only likely medicine names
#         filtered_medicines = []
#         for medicine in medicines:
#             # Remove dosage information for cleaner names
#             clean_name = re.sub(r'\s*\d+\s*mg.*', '', medicine)
#             if clean_name and clean_name not in exclusions:
#                 filtered_medicines.append(clean_name)
        
#         # Remove duplicates and sort
#         return list(set(filtered_medicines))
    
#     def generate_medicine_explanation(self, medicine_name):
#         """Generate explanation for a specific medicine"""
#         # Fallback explanations for common medicines
#         medicine_info = {
#             "paracetamol": {
#                 "name": "Paracetamol (Acetaminophen)",
#                 "use": "Pain relief and fever reduction",
#                 "dosage": "Usually 500-1000mg every 4-6 hours, maximum 4g per day",
#                 "side_effects": "Generally safe when used as directed. Overdose can cause liver damage."
#             },
#             "ibuprofen": {
#                 "name": "Ibuprofen",
#                 "use": "Pain relief, inflammation reduction, and fever reduction",
#                 "dosage": "Usually 200-400mg every 4-6 hours with food",
#                 "side_effects": "May cause stomach upset, increased bleeding risk, kidney issues with long-term use."
#             },
#             "amoxicillin": {
#                 "name": "Amoxicillin",
#                 "use": "Antibiotic for bacterial infections",
#                 "dosage": "Usually 250-500mg every 8 hours, complete the full course",
#                 "side_effects": "May cause diarrhea, nausea, allergic reactions. Do not stop early."
#             },
#             "aspirin": {
#                 "name": "Aspirin",
#                 "use": "Pain relief, fever reduction, blood thinning",
#                 "dosage": "75-300mg daily for blood thinning, 300-600mg for pain relief",
#                 "side_effects": "May cause stomach bleeding, increased bleeding risk."
#             }
#         }
        
#         medicine_lower = medicine_name.lower()
        
#         # Check if we have information for this medicine
#         for key, info in medicine_info.items():
#             if key in medicine_lower or medicine_lower in key:
#                 return f"""
# **{info['name']}**

# **What it's used for:** {info['use']}

# **How to take it:** {info['dosage']}

# **Important notes:** {info['side_effects']}

# Please consult your doctor or pharmacist for personalized advice.
# """
        
#         # Try to use AI model if available
#         if self.explanation_model:
#             try:
#                 prompt = f"Explain the medicine {medicine_name}: what it's used for, how to take it, and important side effects."
#                 response = self.explanation_model(prompt, max_length=200, num_return_sequences=1)
#                 return response[0]['generated_text'].replace(prompt, "").strip()
#             except Exception as e:
#                 print(f"Error generating explanation: {e}")
        
#         # Default explanation
#         return f"""
# **{medicine_name}**

# This appears to be a prescribed medication. Please consult your doctor or pharmacist for detailed information about:

# - What this medicine is used for
# - How and when to take it
# - Possible side effects
# - Drug interactions

# Always follow your doctor's instructions and read the patient information leaflet.
# """
    
#     def translate_to_persian(self, text):
#         """Translate text to Persian"""
#         if not self.translation_model or not self.translation_tokenizer:
#             return self._fallback_persian_translation(text)
        
#         try:
#             # Split long text into chunks
#             sentences = text.split('.')
#             translated_sentences = []
            
#             for sentence in sentences:
#                 if sentence.strip():
#                     inputs = self.translation_tokenizer.encode(sentence.strip(), return_tensors="pt", truncation=True, max_length=512)
#                     if self.device == "cuda":
#                         inputs = inputs.cuda()
                    
#                     with torch.no_grad():
#                         outputs = self.translation_model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
                    
#                     translated = self.translation_tokenizer.decode(outputs[0], skip_special_tokens=True)
#                     translated_sentences.append(translated)
            
#             return '. '.join(translated_sentences)
            
#         except Exception as e:
#             print(f"Translation error: {e}")
#             return self._fallback_persian_translation(text)
    
#     def _fallback_persian_translation(self, text):
#         """Fallback Persian translation using basic dictionary lookup"""
#         # Common medical terms translation dictionary
#         translation_dict = {
#             # Medicine names
#             'paracetamol': 'پارسیتامول',
#             'acetaminophen': 'استامینوفن',
#             'ibuprofen': 'ایبوپروفن',
#             'amoxicillin': 'آموکسی‌سیلین',
#             'aspirin': 'آسپیرین',
            
#             # Medical terms
#             'pain relief': 'کاهش درد',
#             'fever reduction': 'کاهش تب',
#             'antibiotic': 'آنتی‌بیوتیک',
#             'anti-inflammatory': 'ضد التهاب',
#             'side effects': 'عوارض جانبی',
#             'dosage': 'دوز مصرف',
#             'tablet': 'قرص',
#             'capsule': 'کپسول',
#             'daily': 'روزانه',
#             'twice daily': 'دو بار در روز',
#             'three times daily': 'سه بار در روز',
#             'every 6 hours': 'هر ۶ ساعت',
#             'with food': 'همراه با غذا',
#             'after meals': 'بعد از غذا',
#             'doctor': 'پزشک',
#             'pharmacist': 'داروساز',
#             'medicine': 'دارو',
#             'medication': 'دارو',
#             'prescription': 'نسخه',
#             'take': 'مصرف کنید',
#             'used for': 'برای استفاده',
#             'liver damage': 'آسیب کبدی',
#             'stomach upset': 'ناراحتی معده',
#             'allergic reactions': 'واکنش‌های آلرژیک',
#             'nausea': 'حالت تهوع',
#             'diarrhea': 'اسهال',
#             'bleeding risk': 'خطر خونریزی',
#             'kidney issues': 'مشکلات کلیوی',
#             'blood thinning': 'رقیق کردن خون',
#             'bacterial infections': 'عفونت‌های باکتریایی',
#             'complete the full course': 'دوره کامل درمان را تمام کنید',
#             'consult': 'مشورت کنید',
#             'healthcare provider': 'ارائه‌دهنده مراقبت‌های بهداشتی',
#             'as needed': 'در صورت نیاز',
#             'what it\'s used for': 'موارد استفاده',
#             'how to take it': 'نحوه مصرف',
#             'important notes': 'نکات مهم',
#             'usually': 'معمولاً',
#             'maximum': 'حداکثر',
#             'per day': 'در روز',
#             'may cause': 'ممکن است باعث شود',
#             'generally safe': 'معمولاً ایمن',
#             'when used as directed': 'در صورت استفاده طبق دستور',
#             'overdose': 'مصرف بیش از حد',
#             'can cause': 'می‌تواند باعث شود',
#             'inflammation reduction': 'کاهش التهاب',
#             'every 4-6 hours': 'هر ۴-۶ ساعت',
#             'every 8 hours': 'هر ۸ ساعت',
#             'do not stop early': 'زودتر قطع نکنید',
#             'please consult your doctor or pharmacist for personalized advice': 'لطفاً برای مشاوره شخصی‌سازی شده با پزشک یا داروساز مشورت کنید'
#         }
        
#         # First, try to translate longer phrases
#         translated_text = text.lower()
#         for en_phrase, fa_phrase in sorted(translation_dict.items(), key=len, reverse=True):
#             if en_phrase in translated_text:
#                 translated_text = translated_text.replace(en_phrase, fa_phrase)
        
#         # If we still have a lot of English text, provide a structured Persian translation
#         persian_chars = sum(1 for c in translated_text if '\u0600' <= c <= '\u06FF')
#         total_chars = len(translated_text.replace(' ', ''))
        
#         if persian_chars < total_chars * 0.3:  # Less than 30% Persian
#             # Extract medicine names from the original text for structured translation
#             medicine_names = []
#             for key, value in translation_dict.items():
#                 if any(name in text.lower() for name in ['paracetamol', 'ibuprofen', 'amoxicillin', 'aspirin']):
#                     if key in ['paracetamol', 'ibuprofen', 'amoxicillin', 'aspirin'] and key in text.lower():
#                         medicine_names.append(value)
            
#             structured_translation = """
# توضیحات داروها:

# """
#             if 'paracetamol' in text.lower():
#                 structured_translation += """
# **پارسیتامول:**
# • موارد استفاده: کاهش درد و تب
# • نحوه مصرف: معمولاً ۵۰۰-۱۰۰۰ میلی‌گرم هر ۴-۶ ساعت، حداکثر ۴ گرم در روز
# • نکات مهم: در صورت استفاده طبق دستور ایمن است. مصرف بیش از حد می‌تواند باعث آسیب کبدی شود.

# """
            
#             if 'ibuprofen' in text.lower():
#                 structured_translation += """
# **ایبوپروفن:**
# • موارد استفاده: کاهش درد، التهاب و تب
# • نحوه مصرف: معمولاً ۲۰۰-۴۰۰ میلی‌گرم هر ۴-۶ ساعت همراه با غذا
# • نکات مهم: ممکن است باعث ناراحتی معده، افزایش خطر خونریزی و مشکلات کلیوی شود.

# """
            
#             if 'amoxicillin' in text.lower():
#                 structured_translation += """
# **آموکسی‌سیلین:**
# • موارد استفاده: آنتی‌بیوتیک برای عفونت‌های باکتریایی
# • نحوه مصرف: معمولاً ۲۵۰-۵۰۰ میلی‌گرم هر ۸ ساعت
# • نکات مهم: ممکن است باعث اسهال، حالت تهوع و واکنش‌های آلرژیک شود. دوره کامل درمان را تمام کنید.

# """
            
#             structured_translation += """
# **توصیه مهم:** لطفاً برای مشاوره شخصی‌سازی شده با پزشک یا داروساز مشورت کنید.
# """
            
#             return structured_translation
        
#         return translated_text

# # Global instance
# medicine_explainer = MedicineExplainer()

# class PrescriptionReaderView(APIView):
#     parser_classes = (MultiPartParser, FormParser)
    
#     def post(self, request, *args, **kwargs):
#         try:
#             # Get uploaded image
#             if 'image' not in request.FILES:
#                 return Response(
#                     {'error': 'No image file provided'}, 
#                     status=status.HTTP_400_BAD_REQUEST
#                 )
            
#             image_file = request.FILES['image']
            
#             # Open and process image
#             image = Image.open(image_file)
            
#             # Extract text using OCR
#             extracted_text = pytesseract.image_to_string(image)
            
#             if not extracted_text.strip():
#                 return Response(
#                     {'error': 'No text could be extracted from the image'}, 
#                     status=status.HTTP_400_BAD_REQUEST
#                 )
            
#             # Extract medicine names
#             medicines = medicine_explainer.extract_medicine_names(extracted_text)
            
#             # Generate explanations for each medicine
#             explanations = []
#             for medicine in medicines[:5]:  # Limit to 5 medicines to avoid long responses
#                 explanation = medicine_explainer.generate_medicine_explanation(medicine)
#                 explanations.append(f"**{medicine}:**\n{explanation}")
            
#             # Combine explanations
#             full_explanation = "\n\n".join(explanations)
            
#             if not explanations:
#                 full_explanation = """
# No specific medicines were clearly identified in the prescription. 

# **General Advice:**
# - Take all medications exactly as prescribed by your doctor
# - Read the patient information leaflet for each medicine
# - Ask your pharmacist if you have any questions
# - Do not stop taking medications without consulting your doctor
# - Report any side effects to your healthcare provider
# """
            
#             # Translate to Persian
#             persian_explanation = medicine_explainer.translate_to_persian(full_explanation)
            
#             # Return structured response
#             response_data = {
#                 'extracted_text': extracted_text,
#                 'medicines_found': medicines,
#                 'explanation_en': full_explanation,
#                 'explanation_fa': persian_explanation,
#                 'status': 'success'
#             }
            
#             return Response(response_data, status=status.HTTP_200_OK)
            
#         except Exception as e:
#             return Response(
#                 {'error': f'An error occurred: {str(e)}'}, 
#                 status=status.HTTP_500_INTERNAL_SERVER_ERROR
#             )

# class PrescriptionFormView(TemplateView):
#     template_name = 'prescription_form.html'
    
#     def get_context_data(self, **kwargs):
#         context = super().get_context_data(**kwargs)
#         context['title'] = 'Prescription Reader'
#         return context
