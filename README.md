# 🏥 MediScan Pro - AI-Powered Medical Prescription Analysis

A comprehensive full-stack web application that uses advanced OCR and AI to analyze medical prescriptions and provide bilingual medicine explanations in English and Persian/Dari using **Google's Flan-T5-base model**.

## 🌟 Key Features

- **📤 Smart Upload & 📸 Camera Capture**: Upload prescription images or take photos directly
- **🤖 Advanced OCR**: Extract text from prescription images using Tesseract OCR with OpenCV preprocessing
- **🧠 AI Medicine Detection**: Identify 100+ common medicines using pattern recognition and medical databases
- **💊 Google Flan-T5 Explanations**: Generate comprehensive medicine explanations using Google's Flan-T5-base model
- **🌐 Bilingual Support**: Automatic translation to Persian/Dari using Helsinki-NLP neural translation
- **🔍 Intelligent Search**: Search for specific medicines and get instant AI-generated bilingual information
- **📱 Responsive Design**: Modern UI built with Bootstrap 5.3 + Tailwind CSS
- **💾 Data Management**: SQLite database with comprehensive models for tracking uploads and searches

## 🛠️ Technology Stack

### Backend
- **Django 5.2.5** - Web framework
- **Django REST Framework** - API development
- **Python 3.13** - Programming language
- **SQLite** - Database (easily upgradeable to PostgreSQL)

### AI/ML Components
- **🤖 Google Flan-T5-base** - Medicine explanation generation (Primary AI model)
- **Tesseract OCR** - Text extraction from images
- **Helsinki-NLP/opus-mt-en-fa** - English to Persian/Dari translation
- **OpenCV** - Advanced image preprocessing
- **PyTorch & Transformers** - ML model inference

### Frontend
- **Bootstrap 5.3** - UI framework
- **Tailwind CSS** - Utility-first CSS
- **JavaScript ES6+** - Interactive functionality
- **Font Awesome** - Icons
- **Responsive Design** - Mobile-first approach

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ (tested with Python 3.13)
- pip (Python package manager)
- Tesseract OCR (for text extraction)

### Installation Steps

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd medical-prescription-app
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR**
   
   **Ubuntu/Debian:**
   ```bash
   sudo apt-get install tesseract-ocr
   ```
   
   **macOS:**
   ```bash
   brew install tesseract
   ```
   
   **Windows:**
   Download from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

4. **Setup Database**
   ```bash
   python manage.py migrate
   python manage.py createsuperuser  # Optional
   ```

5. **Launch Application**
   ```bash
   python manage.py runserver
   ```

6. **Access Application**
   Open `http://localhost:8000` in your browser

## 🏗️ Project Architecture

```
medical-prescription-app/
├── 🏢 Theranous/                 # Django project configuration
│   ├── settings.py              # Main settings with AI model config
│   ├── urls.py                  # URL routing
│   └── wsgi.py                  # WSGI application
├── 💊 prescription_api/         # Core Django app
│   ├── 🗄️ models.py             # Database models (4 comprehensive models)
│   ├── 👁️ views.py              # API views and frontend views
│   ├── 📋 serializers.py        # DRF serializers
│   ├── 🤖 services.py           # AI/ML services (Flan-T5, OCR, Translation)
│   ├── 🔗 urls.py               # App URL routing
│   └── ⚙️ admin.py              # Django admin configuration
├── �� templates/               # HTML templates
│   ├── base.html               # Bootstrap + Tailwind base template
│   ├── index.html              # Homepage with features
│   ├── upload_prescription.html # Upload page with camera support
│   └── search_medicine.html    # Search page with AI integration
├── 📁 static/                  # Static files
├── 📸 media/                   # User uploaded files
├── 📋 requirements.txt         # Python dependencies
└── 📖 README.md               # This documentation
```

## 🔌 API Endpoints

### Core AI-Powered Endpoints

| Endpoint | Method | Description | AI Integration |
|----------|--------|-------------|----------------|
| `/api/upload/` | POST | Upload prescription image | - |
| `/api/process/<uuid>/` | POST | **Full AI Pipeline** | OCR + Flan-T5 + Translation |
| `/api/search/` | POST | **AI Medicine Search** | Flan-T5 + Translation |
| `/api/prescriptions/` | GET | List prescriptions | - |
| `/api/prescriptions/<uuid>/` | GET | Get prescription details | - |
| `/api/searches/` | GET | List AI-generated searches | - |
| `/api/health/` | GET | System health check | Model status |

### Frontend Routes

| Route | Description |
|-------|-------------|
| `/` | Homepage with AI features showcase |
| `/upload/` | Prescription upload with AI processing |
| `/search/` | AI-powered medicine search |
| `/admin/` | Django admin interface |

## 🤖 Google Flan-T5 Integration

### Model Configuration
The application uses **Google's Flan-T5-base model** for generating comprehensive medicine explanations:

```python
# In services.py - ExplanationService class
model_name = "google/flan-t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)
```

### Optimized Generation Parameters
```python
outputs = model.generate(
    inputs,
    max_length=300,        # Comprehensive explanations
    min_length=80,         # Detailed minimum content
    num_beams=6,          # Higher quality generation
    temperature=0.3,       # Focused, accurate responses
    repetition_penalty=1.2, # Reduce repetition
    early_stopping=True    # Efficient generation
)
```

### Enhanced Prompting
The system uses sophisticated prompts for better medical explanations:
```python
prompt = f"""Provide a clear, comprehensive explanation about the medication {medicine_name}. 
Include: 1) What it is used for (main purpose), 2) How it works in the body, 
3) Common conditions it treats, 4) Important usage information. 
Write in simple, easy-to-understand language for patients."""
```

## 🧠 AI/ML Pipeline

1. **📸 Image Preprocessing**: OpenCV enhances image quality for better OCR
2. **👁️ OCR Processing**: Tesseract extracts text with custom medical configurations
3. **🔍 Medicine Detection**: Pattern matching + comprehensive medicine database (100+ drugs)
4. **🤖 Flan-T5 Generation**: Google's AI model creates detailed English explanations
5. **🌐 Neural Translation**: Helsinki-NLP translates to Persian/Dari
6. **✨ Post-processing**: Clean, format, and optimize responses

## 📊 Database Models

### PrescriptionUpload
- UUID primary key for security
- Image storage with processing status
- OCR text storage
- Timestamp tracking

### DetectedMedicine
- Linked to prescriptions
- Medicine names with confidence scores
- **Flan-T5 generated explanations** (English)
- **Neural translated explanations** (Persian/Dari)

### MedicineSearch
- User search queries
- **AI-generated responses**
- Bilingual explanation storage
- Analytics tracking

### MedicineInfo
- Pre-stored medicine database
- Common medications reference
- Fallback information system

## 🌐 Usage Examples

### Upload & AI Analysis
1. Navigate to `/upload/`
2. Take photo or upload prescription image
3. **AI Processing**: OCR → Medicine Detection → Flan-T5 Explanation → Translation
4. View comprehensive bilingual results

### AI-Powered Search
1. Navigate to `/search/`
2. Enter medicine name (e.g., "aspirin")
3. **Flan-T5 generates detailed explanation**
4. **Automatic Persian/Dari translation**
5. View bilingual results instantly

### API Integration
```python
import requests

# AI-powered medicine search
data = {'medicine_name': 'aspirin'}
response = requests.post('http://localhost:8000/api/search/', json=data)
ai_explanation = response.json()

print(f"English (Flan-T5): {ai_explanation['explanation_english']}")
print(f"Persian (Helsinki-NLP): {ai_explanation['explanation_persian']}")
```

## ⚡ Performance Optimization

### Model Optimization
- **GPU acceleration** support (CUDA when available)
- **Model caching** for faster subsequent requests
- **Batch processing** for multiple medicines
- **Fallback system** with pre-written explanations

### Production Considerations
```python
# Enable GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimize for production
model.eval()  # Set to evaluation mode
torch.no_grad()  # Disable gradient computation
```

## 🔧 Configuration

### AI Model Settings
```python
# In settings.py - Add these for production
AI_MODELS = {
    'FLAN_T5': {
        'MODEL_NAME': 'google/flan-t5-base',
        'MAX_LENGTH': 300,
        'MIN_LENGTH': 80,
        'NUM_BEAMS': 6,
        'TEMPERATURE': 0.3,
    },
    'TRANSLATION': {
        'MODEL_NAME': 'Helsinki-NLP/opus-mt-en-fa',
        'MAX_LENGTH': 512,
    }
}
```

### Environment Variables
```bash
# Optional environment configuration
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0
export TRANSFORMERS_CACHE=/path/to/cache  # Model cache location
export DEBUG=False  # Production mode
```

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Download AI models (optional pre-loading)
RUN python -c "from transformers import T5ForConditionalGeneration, T5Tokenizer; T5ForConditionalGeneration.from_pretrained('google/flan-t5-base'); T5Tokenizer.from_pretrained('google/flan-t5-base')"

EXPOSE 8000
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
```

### Production Checklist
- [ ] Set `DEBUG=False`
- [ ] Configure `ALLOWED_HOSTS`
- [ ] Use PostgreSQL database
- [ ] Set up Redis for caching
- [ ] Configure HTTPS
- [ ] Pre-load AI models
- [ ] Set up monitoring

## 🧪 Testing the AI Integration

### Test Flan-T5 Model
```bash
python manage.py shell
```

```python
from prescription_api.services import explanation_service

# Test Flan-T5 explanation generation
explanation = explanation_service.generate_explanation("aspirin")
print(f"Flan-T5 Generated: {explanation}")
```

### Test Translation
```python
from prescription_api.services import translation_service

# Test Persian translation
persian = translation_service.translate_to_persian("Aspirin is a pain reliever")
print(f"Translated: {persian}")
```

## ⚠️ Important Notes

### Medical Disclaimer
**This application uses AI for informational purposes only.** Always consult healthcare professionals before taking any medication. The **Flan-T5 generated explanations** should supplement, not replace, professional medical advice.

### AI Model Considerations
- **First-time loading**: Models download automatically (may take 5-10 minutes)
- **Memory requirements**: Flan-T5-base requires ~1GB RAM
- **GPU acceleration**: Significantly improves performance
- **Internet required**: For initial model downloads

### Privacy & Security
- All AI processing happens **locally** (no external API calls)
- Uploaded images stored locally
- **No personal health data** sent to external services
- GDPR-compliant design

## 🐛 Troubleshooting

### Common Issues

**Models not loading:**
```bash
# Clear model cache and retry
rm -rf ~/.cache/huggingface/
python manage.py runserver
```

**Tesseract errors:**
```bash
# Verify installation
tesseract --version
which tesseract
```

**Memory issues:**
```bash
# Use CPU-only mode
export CUDA_VISIBLE_DEVICES=""
```

**Slow AI generation:**
```bash
# Enable GPU acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📈 Future Enhancements

- [ ] **Flan-T5-large** model integration for even better explanations
- [ ] **Multi-language support** (Arabic, Urdu, Hindi)
- [ ] **Drug interaction checking**
- [ ] **Dosage information extraction**
- [ ] **Side effects prediction**
- [ ] **Medical image classification**

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Test AI model integration
4. Submit pull request with model performance metrics

## 📄 License

MIT License - Built with ❤️ for better healthcare accessibility

## 🙏 Acknowledgments

- **Google** for the Flan-T5-base model
- **Helsinki-NLP** for the translation model
- **Hugging Face** for the transformers library
- **Tesseract OCR** community
- **Django** and **Bootstrap** communities

---

**🤖 Powered by Google Flan-T5-base for Superior Medicine Explanations**
