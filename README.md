# 🏥 Theranous - Prescription Reader API

**An enhanced Django API that processes prescription images using OCR and provides detailed medicine explanations in both English and Persian (Farsi).**

## ✨ Features

- **📷 OCR Text Extraction**: Extract text from prescription images using Tesseract
- **🤖 AI Medicine Recognition**: Intelligent extraction and identification of medicine names
- **📖 Detailed Explanations**: Generate comprehensive medicine explanations using Hugging Face models
- **🌍 Persian Translation**: Automatic translation to Persian/Farsi with medical terminology
- **🎨 Modern Web Interface**: Beautiful, responsive UI for easy interaction
- **📱 REST API**: Full REST API with JSON responses

## 🎯 Output Format

The API returns structured JSON responses as requested:

```json
{
  "extracted_text": "Paracetamol 500mg\nTake 1 tablet every 6 hours...",
  "medicines_found": ["Paracetamol", "Amoxicillin", "Ibuprofen"],
  "explanation_en": "**Paracetamol:** Pain relief and fever reduction...",
  "explanation_fa": "**پارسیتامول:** کاهش درد و تب...",
  "status": "success"
}
```

## 🔧 Technology Stack

- **Backend**: Django + Django REST Framework
- **OCR**: Tesseract (pytesseract)
- **AI Models**: 
  - `microsoft/DialoGPT-medium` for medicine explanations
  - Fallback Persian translation with medical dictionary
- **Frontend**: HTML5 + CSS3 + JavaScript
- **Database**: SQLite (default)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd theranous-prescription-reader

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Tesseract OCR
sudo apt-get install tesseract-ocr  # Ubuntu/Debian
# brew install tesseract  # macOS
```

### 2. Run the Server

```bash
# Easy start with auto-setup
python start_server.py

# Or manually:
source venv/bin/activate
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

### 3. Test the System

**Web Interface:**
- Open: http://127.0.0.1:8000/
- Upload a prescription image
- View results in English and Persian

**API Test:**
```bash
curl -X POST http://127.0.0.1:8000/api/prescription/ \
     -F 'image=@prescription.jpg'
```

**Demo Script:**
```bash
python demo_prescription_reader.py
```

## 📡 API Endpoints

### Upload Prescription
- **URL**: `/api/prescription/`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Parameters**: 
  - `image`: Prescription image file

### Web Interface
- **URL**: `/`
- **Method**: `GET`
- **Description**: HTML form for prescription upload

## 🧠 Medicine Recognition

The system uses advanced pattern recognition to identify medicines:

### Supported Patterns
- Medicine names with common suffixes (cillin, mycin, zole, etc.)
- Known medicine database (Paracetamol, Ibuprofen, Amoxicillin, etc.)
- Medicine names with dosage (e.g., "Paracetamol 500mg")

### Excluded Words
- Common prescription text (Doctor, Patient, Date, etc.)
- Medical facility names (Hospital, Clinic, etc.)
- Instructions (Take, Daily, etc.)

## 🌍 Persian Translation

### Features
- Medical terminology dictionary
- Structured translations for common medicines
- Fallback explanations in Persian
- Right-to-left text support

### Supported Medicines
- پارسیتامول (Paracetamol)
- ایبوپروفن (Ibuprofen) 
- آموکسی‌سیلین (Amoxicillin)
- آسپیرین (Aspirin)
- And many more...

## 📖 Example Usage

### Python API Call
```python
import requests

# Upload prescription image
with open('prescription.jpg', 'rb') as f:
    response = requests.post(
        'http://127.0.0.1:8000/api/prescription/',
        files={'image': f}
    )

result = response.json()
print("English:", result['explanation_en'])
print("Persian:", result['explanation_fa'])
```

### JavaScript/AJAX
```javascript
const formData = new FormData();
formData.append('image', fileInput.files[0]);

fetch('/api/prescription/', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('English:', data.explanation_en);
    console.log('Persian:', data.explanation_fa);
});
```

## 🔍 How It Works

1. **Image Upload**: User uploads prescription image via web interface or API
2. **OCR Processing**: Tesseract extracts text from the image
3. **Medicine Extraction**: AI identifies medicine names using patterns and databases
4. **Explanation Generation**: Creates detailed explanations for each medicine
5. **Persian Translation**: Translates explanations using medical dictionary
6. **Response**: Returns structured JSON with all information

## ⚕️ Medicine Information Provided

For each identified medicine:

### English Explanation
- **Correct medicine name** (if misspelled in OCR)
- **What it's used for** (indications)
- **How and when to take it** (dosage and timing)
- **Common side effects** and warnings
- **Important notes** and precautions

### Persian Translation
- **نام دارو** (Medicine name)
- **موارد استفاده** (Usage indications)
- **نحوه مصرف** (How to take)
- **عوارض جانبی** (Side effects)
- **نکات مهم** (Important notes)

## 🛡️ Safety Features

- **Medical Disclaimers**: All responses include advice to consult healthcare providers
- **Error Handling**: Graceful handling of OCR failures and missing medicines
- **Input Validation**: Validates image uploads and handles errors
- **Fallback Responses**: Provides generic advice when specific information unavailable

## 📁 Project Structure

```
theranous/
├── api/                    # Django app
│   ├── views.py           # Main API logic
│   ├── urls.py            # URL routing
│   └── models.py          # Database models
├── templates/             # HTML templates
│   └── prescription_form.html
├── Theranous/            # Django project
│   ├── settings.py       # Configuration
│   └── urls.py           # Main URL config
├── requirements.txt      # Dependencies
├── demo_prescription_reader.py  # Demo script
├── start_server.py       # Easy startup
└── README.md            # This file
```

## 🐛 Troubleshooting

### Common Issues

**OCR not working:**
```bash
# Install Tesseract
sudo apt-get install tesseract-ocr
```

**Translation model failed:**
- System uses fallback Persian translation
- Add API keys for better translation services if needed

**Virtual environment issues:**
```bash
# Recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Tesseract OCR** for text extraction
- **Hugging Face** for AI models
- **Django REST Framework** for API development
- **Bootstrap** for UI components

## 📞 Support

For support, please create an issue in the repository or contact the development team.

---

**⚕️ Medical Disclaimer**: This tool is for informational purposes only and should not replace professional medical advice. Always consult with healthcare providers for medical decisions.
