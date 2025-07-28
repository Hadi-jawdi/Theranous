#!/usr/bin/env python3
"""
Test script for the medicine explanation functionality
"""

import os
import sys
import django

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Theranous.settings')
django.setup()

from api.views import MedicineExplainer

def test_medicine_explanation():
    """Test the medicine explanation and translation functionality"""
    
    print("🧪 Testing Medicine Explanation System...")
    
    # Test sample prescription text
    sample_text = """
    Paracetamol 500mg
    Take 1 tablet every 6 hours
    
    Amoxicillin 250mg
    Take 1 capsule twice daily
    
    Ibuprofen 400mg
    Take as needed for pain
    """
    
    print(f"📝 Sample OCR Text:\n{sample_text}")
    
    # Initialize the explainer
    try:
        print("\n🔄 Initializing Medicine Explainer...")
        explainer = MedicineExplainer()
        print("✅ Medicine Explainer initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Test medicine extraction
    print("\n🔍 Extracting medicine names...")
    try:
        medicines = explainer.extract_medicine_names(sample_text)
        print(f"Extracted medicines: {medicines}")
    except Exception as e:
        print(f"❌ Error extracting medicines: {e}")
        return
    
    # Test explanation generation
    print("\n📖 Generating explanations...")
    try:
        explanations = []
        for medicine in medicines[:3]:  # Test first 3 medicines
            explanation = explainer.generate_medicine_explanation(medicine)
            explanations.append(f"**{medicine}:**\n{explanation}")
            print(f"\n✅ Generated explanation for {medicine}")
        
        full_explanation = "\n\n".join(explanations)
        print(f"\nEnglish Explanation:\n{full_explanation[:500]}...")
        
    except Exception as e:
        print(f"❌ Error generating explanation: {e}")
        return
    
    # Test translation
    print("\n🌍 Testing Persian translation...")
    try:
        persian_explanation = explainer.translate_to_persian("Paracetamol is used for pain relief and fever reduction.")
        print(f"Persian Translation:\n{persian_explanation}")
    except Exception as e:
        print(f"❌ Translation error: {e}")
    
    print("\n✅ Test completed!")

def test_api_endpoint():
    """Test the API endpoint functionality"""
    print("\n🌐 Testing API Endpoint...")
    
    try:
        from django.test import Client
        from django.core.files.uploadedfile import SimpleUploadedFile
        from PIL import Image
        import io
        
        # Create a test image
        img = Image.new('RGB', (300, 100), color='white')
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Create uploaded file
        uploaded_file = SimpleUploadedFile(
            "test_prescription.png",
            img_buffer.getvalue(),
            content_type="image/png"
        )
        
        # Test the API
        client = Client()
        response = client.post('/api/prescription/', {'image': uploaded_file})
        
        print(f"API Response Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ API endpoint working correctly")
        else:
            print(f"❌ API error: {response.content}")
            
    except Exception as e:
        print(f"❌ API test error: {e}")

if __name__ == "__main__":
    test_medicine_explanation()
    test_api_endpoint()