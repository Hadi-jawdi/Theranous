from django.contrib import admin
from .models import PrescriptionUpload, DetectedMedicine, MedicineSearch, MedicineInfo

@admin.register(PrescriptionUpload)
class PrescriptionUploadAdmin(admin.ModelAdmin):
    list_display = ['id', 'uploaded_at', 'processed_at', 'has_ocr_text']
    list_filter = ['uploaded_at', 'processed_at']
    search_fields = ['id', 'ocr_text']
    readonly_fields = ['id', 'uploaded_at']
    
    def has_ocr_text(self, obj):
        return bool(obj.ocr_text)
    has_ocr_text.boolean = True
    has_ocr_text.short_description = 'OCR Processed'

@admin.register(DetectedMedicine)
class DetectedMedicineAdmin(admin.ModelAdmin):
    list_display = ['medicine_name', 'prescription', 'confidence_score', 'created_at']
    list_filter = ['created_at', 'confidence_score']
    search_fields = ['medicine_name', 'prescription__id']
    readonly_fields = ['created_at']

@admin.register(MedicineSearch)
class MedicineSearchAdmin(admin.ModelAdmin):
    list_display = ['search_query', 'medicine_name', 'searched_at', 'ip_address']
    list_filter = ['searched_at']
    search_fields = ['search_query', 'medicine_name']
    readonly_fields = ['id', 'searched_at']

@admin.register(MedicineInfo)
class MedicineInfoAdmin(admin.ModelAdmin):
    list_display = ['name', 'generic_name', 'category', 'created_at']
    list_filter = ['category', 'created_at']
    search_fields = ['name', 'generic_name', 'category']
    readonly_fields = ['created_at', 'updated_at']
