from django.urls import path
from . import views, api_views

urlpatterns = [
    # path('',views.index),
    path('', views.DiagnosisPatientStatus.as_view()), #Normal HTML View
    path('save', views.SaveToDataset.as_view()), #Normal HTML View
    path('api', api_views.PatientStatusApiView.as_view()), #Api End Point View
]