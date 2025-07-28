from django.urls import path
from . import views

urlpatterns = [
    path('', views.PrescriptionFormView.as_view(), name='prescription_form'),
    path('prescription/', views.PrescriptionReaderView.as_view(), name='prescription_reader'),
] 