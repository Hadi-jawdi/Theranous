from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
import pytesseract
from PIL import Image
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, MarianMTModel, MarianTokenizer
import torch
import io
from django.views.generic import TemplateView
from django.http import JsonResponse

# Create your views here.

# --- API Endpoint for Prescription Upload ---
class UploadPrescriptionView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, format=None):
        try:
            image_file = request.FILES.get('image')
            if not image_file:
                return JsonResponse({'error': 'No image uploaded.'}, status=400)

            image = Image.open(image_file)
            extracted_text = pytesseract.image_to_string(image)

            # Generate explanation using BioGPT
            try:
                tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
                model = AutoModelForCausalLM.from_pretrained("microsoft/biogpt")
                input_ids = tokenizer.encode(extracted_text, return_tensors="pt")
                output = model.generate(input_ids, max_length=100, num_return_sequences=1)
                explanation_en = tokenizer.decode(output[0], skip_special_tokens=True)
            except Exception as e:
                explanation_en = f"[BioGPT error: {str(e)}]"

            # Translate explanation to Dari using MarianMT
            try:
                mt_model_name = "Helsinki-NLP/opus-mt-en-fa"
                mt_tokenizer = MarianTokenizer.from_pretrained(mt_model_name)
                mt_model = MarianMTModel.from_pretrained(mt_model_name)
                translated = mt_model.generate(**mt_tokenizer(explanation_en, return_tensors="pt", padding=True))
                explanation_fa = mt_tokenizer.decode(translated[0], skip_special_tokens=True)
            except Exception as e:
                explanation_fa = f"[Translation error: {str(e)}]"

            return JsonResponse({
                'extracted_text': extracted_text,
                'explanation_en': explanation_en,
                'explanation_fa': explanation_fa
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

# --- Frontend View for Upload Form ---
class PrescriptionFormView(TemplateView):
    template_name = "prescription_form.html"

    def get(self, request, *args, **kwargs):
        # Render the upload form HTML
        return render(request, self.template_name)
