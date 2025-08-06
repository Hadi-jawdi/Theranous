from django.urls import path
from . import views

app_name = 'prescription_api'

urlpatterns = [
    # API endpoints
    path('api/upload/', views.PrescriptionUploadView.as_view(), name='upload_prescription'),
    path('api/process/<uuid:prescription_id>/', views.ProcessPrescriptionView.as_view(), name='process_prescription'),
    path('api/search/', views.MedicineSearchView.as_view(), name='search_medicine'),
    path('api/prescriptions/', views.PrescriptionListView.as_view(), name='prescription_list'),
    path('api/prescriptions/<uuid:pk>/', views.PrescriptionDetailView.as_view(), name='prescription_detail'),
    path('api/searches/', views.MedicineSearchListView.as_view(), name='search_list'),
    path('api/medicines/', views.MedicineInfoListView.as_view(), name='medicine_list'),
    path('api/health/', views.health_check, name='health_check'),
    
    # Frontend views
    path('', views.index, name='index'),
    path('upload/', views.upload_prescription, name='upload_page'),
    path('search/', views.search_medicine, name='search_page'),
]