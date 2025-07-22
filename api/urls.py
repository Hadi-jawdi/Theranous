from django.urls import path
from . import views

urlpatterns = [
    path('', views.PrescriptionFormView.as_view(), name='prescription-form'),
    path('upload-prescription/', views.UploadPrescriptionView.as_view(), name='upload-prescription'),
] 