from django.shortcuts import render

# Create your views here.
from .forms import CustomerForm
from rest_framework import viewsets
from rest_framework.decorators import api_view
from django.core import serializers
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from rest_framework.parsers import JSONParser
from .models import Customer
from .serializers import CustomerSerializers

import pickle
import os
from django.conf import settings
from sklearn import preprocessing
import pandas as pd
from django.shortcuts import render, redirect
from django.contrib import messages

class CustomerView(viewsets.ModelViewSet):
    queryset = Customer.objects.all()
    serializer_class = CustomerSerializers


def status(df):
    try:
        path_to_model = os.path.join(settings.BASE_DIR, 'static/MlModels/')
        scaler = pickle.load(open(path_to_model + "TrainedScaler.sav", 'rb'))
        model = pickle.load(open(path_to_model + "PredictionModel.sav", 'rb'))
        X = scaler.transform(df)
        y_pred = model.predict(X)
        y_pred = (y_pred > 0.80)
        result = "Yes" if y_pred else "No"
        return result
    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)


def FormView(request):
    if request.method == 'POST':
        form = CustomerForm(request.POST or None)

        if form.is_valid():
            Gender = form.cleaned_data['gender']
            Age = form.cleaned_data['age']
            EstimatedSalary = form.cleaned_data['salary']
            df = pd.DataFrame({'Gender': [Gender], 'Age': [Age], 'EstimatedSalary': [EstimatedSalary]})
            df["Gender"] = 1 if "male" else 2
            result = status(df)
            return render(request, 'status.html', {"data": result})

    form = CustomerForm()
    return render(request, 'form.html', {'form': form})