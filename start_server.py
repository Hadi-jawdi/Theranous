#!/usr/bin/env python3

"""
Startup script for Theranous Prescription Reader API
"""

import os
import sys
import subprocess
import time

def start_server():
    """Start the Django development server"""
    
    print("🏥 Starting Theranous Prescription Reader API Server...")
    print("=" * 50)
    
    # Check if virtual environment is activated
    if not os.path.exists('venv'):
        print("❌ Virtual environment not found. Please run:")
        print("   python -m venv venv")
        print("   source venv/bin/activate")
        print("   pip install -r requirements.txt")
        return
    
    print("✅ Virtual environment found")
    
    # Run migrations
    print("\n📊 Running database migrations...")
    try:
        subprocess.run(['python', 'manage.py', 'migrate'], check=True)
        print("✅ Migrations completed")
    except subprocess.CalledProcessError:
        print("❌ Migration failed")
        return
    
    print("\n🚀 Starting Django development server...")
    print("\n📝 API Endpoints:")
    print("   • Web Interface: http://127.0.0.1:8000/")
    print("   • API Endpoint:  http://127.0.0.1:8000/api/prescription/")
    
    print("\n📖 How to test:")
    print("   1. Open browser: http://127.0.0.1:8000/")
    print("   2. Upload a prescription image")
    print("   3. View English and Persian explanations")
    
    print("\n📡 API Usage:")
    print("   curl -X POST http://127.0.0.1:8000/api/prescription/ \\")
    print("        -F 'image=@prescription.jpg'")
    
    print("\n🔧 Demo script:")
    print("   python demo_prescription_reader.py")
    
    print("\n" + "=" * 50)
    print("🌟 Server starting on http://127.0.0.1:8000/")
    print("   Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the server
    try:
        subprocess.run(['python', 'manage.py', 'runserver', '0.0.0.0:8000'])
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Thank you for using Theranous!")

if __name__ == "__main__":
    start_server()