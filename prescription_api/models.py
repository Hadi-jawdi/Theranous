from django.db import models
from django.utils import timezone
import uuid

class PrescriptionUpload(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    uploaded_image = models.ImageField(upload_to='prescriptions/')
    uploaded_at = models.DateTimeField(default=timezone.now)
    ocr_text = models.TextField(blank=True, null=True)
    processed_at = models.DateTimeField(blank=True, null=True)
    
    def __str__(self):
        return f"Prescription {self.id} - {self.uploaded_at.strftime('%Y-%m-%d %H:%M')}"

class DetectedMedicine(models.Model):
    prescription = models.ForeignKey(PrescriptionUpload, on_delete=models.CASCADE, related_name='medicines')
    medicine_name = models.CharField(max_length=200)
    confidence_score = models.FloatField(default=0.0)
    explanation_english = models.TextField(blank=True, null=True)
    explanation_persian = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)
    
    def __str__(self):
        return f"{self.medicine_name} - {self.prescription.id}"

class MedicineSearch(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    search_query = models.CharField(max_length=200)
    medicine_name = models.CharField(max_length=200)
    explanation_english = models.TextField()
    explanation_persian = models.TextField()
    searched_at = models.DateTimeField(default=timezone.now)
    ip_address = models.GenericIPAddressField(blank=True, null=True)
    
    def __str__(self):
        return f"Search: {self.search_query} - {self.searched_at.strftime('%Y-%m-%d %H:%M')}"

class MedicineInfo(models.Model):
    """Pre-stored medicine information for common medications"""
    name = models.CharField(max_length=200, unique=True)
    generic_name = models.CharField(max_length=200, blank=True, null=True)
    category = models.CharField(max_length=100, blank=True, null=True)
    common_uses = models.TextField(blank=True, null=True)
    side_effects = models.TextField(blank=True, null=True)
    dosage_info = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name_plural = "Medicine Information"
    
    def __str__(self):
        return self.name
