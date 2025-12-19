# Theranous - Prescription Reader & Explainer

A Django-based web application that allows users to upload prescription images and receive simple, easy-to-understand explanations in both English and Persian (Farsi).

## Features

- 📸 **Image Upload**: Upload prescription images (JPG, PNG, GIF)
- 🔍 **OCR Text Extraction**: Automatically extracts text from prescription images using Tesseract OCR
- 📝 **Simple Explanations**: Generates user-friendly explanations of prescriptions in plain language
- 🌐 **Bilingual Support**: Provides explanations in both English and Persian (Farsi)
- 💊 **Medication Information**: Identifies medications, dosages, and instructions

## Requirements

- Python 3.8 or higher
- Tesseract OCR installed on your system
  - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki) or use `choco install tesseract`
  - **Linux**: `sudo apt-get install tesseract-ocr` (Ubuntu/Debian) or `sudo yum install tesseract` (CentOS/RHEL)
  - **macOS**: `brew install tesseract`

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Tesseract path** (if needed):
   - On Windows, you may need to set the Tesseract path in your environment or in the code
   - The default installation path is usually: `C:\Program Files\Tesseract-OCR\tesseract.exe`
   - You can set it in your code if needed:
     ```python
     import pytesseract
     pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
     ```

5. **Run database migrations**:
   ```bash
   python manage.py migrate
   ```

6. **Create a superuser** (optional, for admin access):
   ```bash
   python manage.py createsuperuser
   ```

7. **Run the development server**:
   ```bash
   python manage.py runserver
   ```

8. **Access the application**:
   - Open your browser and navigate to: `http://127.0.0.1:8000/api/`

## Usage

1. **Upload a Prescription**:
   - Click on "Select Prescription Image" and choose an image file
   - Click "Upload & Analyze Prescription"

2. **View Results**:
   - The extracted text from the prescription will be displayed
   - Simple explanations will be shown in both English and Persian
   - Review the medication information and instructions

## Technical Details

### Models Used

- **OCR**: Tesseract OCR for text extraction from images
- **Translation**: Helsinki-NLP's OPUS-MT model for English to Persian translation
- **Text Processing**: Custom logic for generating simple, user-friendly explanations

### Model Caching

The translation models are cached in memory to avoid reloading on every request, improving performance.

### API Endpoints

- `GET /api/`: Prescription upload form
- `POST /api/upload-prescription/`: API endpoint for uploading and processing prescriptions

## Notes

- The first request may take longer as models are downloaded and loaded
- Ensure prescription images are clear and well-lit for best OCR results
- The application is designed for development use. For production, consider:
  - Using environment variables for sensitive settings
  - Implementing proper error logging
  - Adding authentication and authorization
  - Setting up proper static file serving
  - Using a production-grade WSGI server

## License

This project is for educational and personal use.


