import os
import time
from django.shortcuts import render
from django.utils import timezone
from django.http import JsonResponse
from rest_framework import status, generics
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from .models import PrescriptionUpload, DetectedMedicine, MedicineSearch, MedicineInfo
from .serializers import (
    PrescriptionUploadSerializer, PrescriptionUploadCreateSerializer,
    MedicineSearchSerializer, MedicineSearchCreateSerializer,
    MedicineInfoSerializer, OCRResultSerializer, MedicineExplanationSerializer,
    ErrorResponseSerializer
)
from .services import (
    ocr_service, medicine_detection_service, 
    explanation_service, translation_service
)

def get_client_ip(request):
    """Get client IP address from request"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

@method_decorator(csrf_exempt, name='dispatch')
class PrescriptionUploadView(APIView):
    """API view for uploading prescription images"""
    permission_classes = [AllowAny]
    parser_classes = [MultiPartParser, FormParser]
    
    def post(self, request, *args, **kwargs):
        serializer = PrescriptionUploadCreateSerializer(data=request.data)
        if serializer.is_valid():
            prescription = serializer.save()
            
            # Return the uploaded prescription info
            response_serializer = PrescriptionUploadSerializer(prescription)
            return Response(response_serializer.data, status=status.HTTP_201_CREATED)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@method_decorator(csrf_exempt, name='dispatch')
class ProcessPrescriptionView(APIView):
    """API view for processing prescription images with OCR and medicine detection"""
    permission_classes = [AllowAny]
    
    def post(self, request, prescription_id):
        try:
            start_time = time.time()
            
            # Get the prescription
            try:
                prescription = PrescriptionUpload.objects.get(id=prescription_id)
            except PrescriptionUpload.DoesNotExist:
                return Response(
                    {'error': 'Prescription not found'}, 
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Get the image path
            image_path = prescription.uploaded_image.path
            
            # Extract text using OCR
            ocr_text = ocr_service.extract_text(image_path)
            
            # Update prescription with OCR text
            prescription.ocr_text = ocr_text
            prescription.processed_at = timezone.now()
            prescription.save()
            
            # Detect medicines in the OCR text
            detected_medicines_data = medicine_detection_service.detect_medicines(ocr_text)
            
            # Create DetectedMedicine objects and generate explanations
            detected_medicines = []
            for medicine_data in detected_medicines_data:
                medicine_name = medicine_data['name']
                confidence = medicine_data['confidence']
                
                # Generate English explanation
                explanation_english = explanation_service.generate_explanation(medicine_name)
                
                # Translate to Persian
                explanation_persian = translation_service.translate_to_persian(explanation_english)
                
                # Create DetectedMedicine object
                detected_medicine = DetectedMedicine.objects.create(
                    prescription=prescription,
                    medicine_name=medicine_name,
                    confidence_score=confidence,
                    explanation_english=explanation_english,
                    explanation_persian=explanation_persian
                )
                detected_medicines.append(detected_medicine)
            
            processing_time = time.time() - start_time
            
            # Prepare response
            from .serializers import DetectedMedicineSerializer
            medicines_serializer = DetectedMedicineSerializer(detected_medicines, many=True)
            
            response_data = {
                'prescription_id': prescription.id,
                'ocr_text': ocr_text,
                'detected_medicines': medicines_serializer.data,
                'processing_time': processing_time
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response(
                {
                    'error': 'Processing failed',
                    'message': str(e)
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

@method_decorator(csrf_exempt, name='dispatch')
class MedicineSearchView(APIView):
    """API view for searching medicine information"""
    permission_classes = [AllowAny]
    
    def post(self, request):
        serializer = MedicineSearchCreateSerializer(data=request.data)
        if serializer.is_valid():
            medicine_name = serializer.validated_data['medicine_name']
            
            try:
                # Generate English explanation
                explanation_english = explanation_service.generate_explanation(medicine_name)
                
                # Translate to Persian
                explanation_persian = translation_service.translate_to_persian(explanation_english)
                
                # Save the search
                search = MedicineSearch.objects.create(
                    search_query=medicine_name,
                    medicine_name=medicine_name,
                    explanation_english=explanation_english,
                    explanation_persian=explanation_persian,
                    ip_address=get_client_ip(request)
                )
                
                response_data = {
                    'medicine_name': medicine_name,
                    'explanation_english': explanation_english,
                    'explanation_persian': explanation_persian,
                    'search_id': search.id
                }
                
                return Response(response_data, status=status.HTTP_200_OK)
                
            except Exception as e:
                return Response(
                    {
                        'error': 'Search failed',
                        'message': str(e)
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class PrescriptionDetailView(generics.RetrieveAPIView):
    """API view for retrieving prescription details"""
    queryset = PrescriptionUpload.objects.all()
    serializer_class = PrescriptionUploadSerializer
    permission_classes = [AllowAny]

class PrescriptionListView(generics.ListAPIView):
    """API view for listing prescriptions"""
    queryset = PrescriptionUpload.objects.all().order_by('-uploaded_at')
    serializer_class = PrescriptionUploadSerializer
    permission_classes = [AllowAny]

class MedicineSearchListView(generics.ListAPIView):
    """API view for listing medicine searches"""
    queryset = MedicineSearch.objects.all().order_by('-searched_at')
    serializer_class = MedicineSearchSerializer
    permission_classes = [AllowAny]

class MedicineInfoListView(generics.ListAPIView):
    """API view for listing medicine information"""
    queryset = MedicineInfo.objects.all().order_by('name')
    serializer_class = MedicineInfoSerializer
    permission_classes = [AllowAny]

@api_view(['GET'])
@permission_classes([AllowAny])
def health_check(request):
    """Health check endpoint"""
    return Response({
        'status': 'healthy',
        'timestamp': timezone.now().isoformat(),
        'services': {
            'ocr': 'available',
            'medicine_detection': 'available',
            'explanation_generation': 'available',
            'translation': 'available'
        }
    })

# Frontend views
def index(request):
    """Main page view"""
    return render(request, 'index.html')

def upload_prescription(request):
    """Upload prescription page view"""
    return render(request, 'upload_prescription.html')

def search_medicine(request):
    """Search medicine page view"""
    return render(request, 'search_medicine.html')
