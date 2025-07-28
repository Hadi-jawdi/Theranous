#!/usr/bin/env python3

"""
Demo script for the Prescription Reader functionality
"""

import os
import sys
import django
import json

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Theranous.settings')
django.setup()

from api.views import MedicineExplainer

def demo_prescription_reader():
    """Demo the complete prescription reading functionality"""
    
    print("🏥 === Theranous Prescription Reader Demo ===\n")
    
    # Sample prescription text (as would be extracted from OCR)
    prescription_text = """
    Dr. Smith Medical Center
    Patient: John Doe
    Date: 2025-01-15
    
    Rx:
    1. Paracetamol 500mg
       Take 1 tablet every 6 hours for pain relief
       Duration: 5 days
    
    2. Amoxicillin 250mg
       Take 1 capsule twice daily
       Duration: 7 days (complete course)
    
    3. Ibuprofen 400mg
       Take as needed for inflammation
       Maximum 3 times daily with food
    
    Dr. Sarah Smith
    License: MD12345
    """
    
    print("📝 Sample Prescription Text:")
    print("=" * 50)
    print(prescription_text)
    print("=" * 50)
    
    # Initialize the medicine explainer
    print("\n🔧 Initializing Medicine Explainer...")
    explainer = MedicineExplainer()
    
    # Extract medicine names
    print("\n🔍 Extracting Medicine Names...")
    medicines = explainer.extract_medicine_names(prescription_text)
    print(f"Found medicines: {medicines}")
    
    # Generate explanations
    print("\n📖 Generating Medicine Explanations...")
    explanations = []
    for medicine in medicines[:3]:  # Process first 3 medicines
        explanation = explainer.generate_medicine_explanation(medicine)
        explanations.append(f"**{medicine}:**\n{explanation}")
        print(f"✅ Generated explanation for: {medicine}")
    
    full_explanation = "\n\n".join(explanations)
    
    # Generate Persian translation
    print("\n🌍 Translating to Persian...")
    persian_explanation = explainer.translate_to_persian(full_explanation)
    
    # Create final response in the expected format
    response = {
        "extracted_text": prescription_text.strip(),
        "medicines_found": medicines,
        "explanation_en": full_explanation,
        "explanation_fa": persian_explanation,
        "status": "success"
    }
    
    # Display results
    print("\n" + "="*60)
    print("📋 FINAL RESULTS")
    print("="*60)
    
    print("\n🔤 ENGLISH EXPLANATION:")
    print("-" * 30)
    print(response["explanation_en"])
    
    print("\n🔤 PERSIAN EXPLANATION:")
    print("-" * 30)
    print(response["explanation_fa"])
    
    print("\n📊 STRUCTURED JSON RESPONSE:")
    print("-" * 30)
    print(json.dumps({
        "extracted_text": response["extracted_text"][:100] + "...",
        "medicines_found": response["medicines_found"],
        "explanation_en": response["explanation_en"][:200] + "...",
        "explanation_fa": "Persian translation provided",
        "status": response["status"]
    }, indent=2, ensure_ascii=False))
    
    print("\n✅ Demo completed successfully!")
    print("\n💡 How to use:")
    print("   1. Upload prescription image to /api/prescription/")
    print("   2. System extracts text using OCR") 
    print("   3. AI identifies medicine names")
    print("   4. Generates explanations in English")
    print("   5. Translates to Persian")
    print("   6. Returns structured JSON response")
    
    return response

if __name__ == "__main__":
    demo_prescription_reader()