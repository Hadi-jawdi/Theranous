from rest_framework import serializers
from .models import PrescriptionUpload, DetectedMedicine, MedicineSearch, MedicineInfo

class DetectedMedicineSerializer(serializers.ModelSerializer):
    class Meta:
        model = DetectedMedicine
        fields = ['id', 'medicine_name', 'confidence_score', 'explanation_english', 
                 'explanation_persian', 'created_at']

class PrescriptionUploadSerializer(serializers.ModelSerializer):
    medicines = DetectedMedicineSerializer(many=True, read_only=True)
    
    class Meta:
        model = PrescriptionUpload
        fields = ['id', 'uploaded_image', 'uploaded_at', 'ocr_text', 
                 'processed_at', 'medicines']

class PrescriptionUploadCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = PrescriptionUpload
        fields = ['uploaded_image']

class MedicineSearchSerializer(serializers.ModelSerializer):
    class Meta:
        model = MedicineSearch
        fields = ['id', 'search_query', 'medicine_name', 'explanation_english', 
                 'explanation_persian', 'searched_at', 'ip_address']

class MedicineSearchCreateSerializer(serializers.Serializer):
    medicine_name = serializers.CharField(max_length=200)

class MedicineInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = MedicineInfo
        fields = ['id', 'name', 'generic_name', 'category', 'common_uses', 
                 'side_effects', 'dosage_info', 'created_at', 'updated_at']

class OCRResultSerializer(serializers.Serializer):
    """Serializer for OCR processing results"""
    prescription_id = serializers.UUIDField()
    ocr_text = serializers.CharField()
    detected_medicines = DetectedMedicineSerializer(many=True)
    processing_time = serializers.FloatField()

class MedicineExplanationSerializer(serializers.Serializer):
    """Serializer for medicine explanation results"""
    medicine_name = serializers.CharField()
    explanation_english = serializers.CharField()
    explanation_persian = serializers.CharField()
    search_id = serializers.UUIDField(required=False)

class ErrorResponseSerializer(serializers.Serializer):
    """Serializer for error responses"""
    error = serializers.CharField()
    message = serializers.CharField()
    details = serializers.DictField(required=False)