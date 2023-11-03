from django.shortcuts import render

# Create your views here.
import joblib
from django.conf import settings
from django.http import HttpResponseRedirect
import numpy as np
import os
from django.views.generic import View, TemplateView
from .forms import PatientForm
import pandas as pd

# Load the model from the static folder
path_to_model = os.path.join(settings.BASE_DIR, 'static/model/')
loaded_model = joblib.load(open(path_to_model + 'diabete_detector_model.pkl', 'rb'))

# For Normal HTML View
class DiagnosisPatientStatus(View):
    def get(self, request):
        form = PatientForm()
        context = {'form': form}
        return render(request, 'form.html', context)

    def post(self, request):
        try:
            form = PatientForm(request.POST)

            if form.is_valid():
                # load the request data
                patient_json_info = form.data
                print(patient_json_info)

                ara = []
                ara.append(float(patient_json_info['pregnancies']))
                ara.append(float(patient_json_info['glucose']))
                ara.append(float(patient_json_info['bloodpressure']))
                ara.append(float(patient_json_info['skinthickness']))
                ara.append(float(patient_json_info['insulin']))
                ara.append(float(patient_json_info['bmi']))
                ara.append(float(patient_json_info['diabetespedigreefunction']))
                ara.append(float(patient_json_info['age']))

                patient_info_float_data = np.array(ara)
                # Retrieve all the values from the json data
                # patient_info = np.array(list(patient_info_float_data.values()))

                # Make prediction
                patient_status = loaded_model.predict([patient_info_float_data])

                # Model confidence score
                model_confidence_score = np.max(loaded_model.predict_proba([patient_info_float_data]))

                model_prediction = {
                    'info': 'success',
                    'patient_status': patient_status[0],
                    'model_confidence_proba': float("{:.2f}".format(model_confidence_score * 100))
                }
                # return model_prediction
                context = {'prediction': model_prediction}
                return render(request, 'result.html', context)


        except ValueError as ve:
            model_prediction = {
                'error_code': '-1',
                "info": str(ve)
            }

        # return model_prediction

        context = {'prediction': model_prediction}
        return render(request, 'result.html', context)

# http://127.0.0.1:8000
# Sample POST Request
#
# {
#     "pregnancies": 6,
#     "glucose": 148,
#     "bloodpressure": 72,
#     "skinthickness": 35,
#     "insulin": 0,
#     "bmi": 33.6,
#     "diabetespedigreefunction": 0.627,
#     "age": 50
# }


class TrainModelView(View):
    def get(self, request):
        form = PatientForm()
        context = {'form': form}
        return render(request, 'form.html', context)

class SaveToDataset(View):
    def post(self, request):
        try:
            form = PatientForm(request.POST)

            if form.is_valid():
                # load the request data
                patient_json_info = form.data
                print(patient_json_info)

                data = {
                    'Pregnancies': [float(patient_json_info['pregnancies'])],
                    'Glucose': [float(patient_json_info['glucose'])],
                    'Bloodpressure': [float(patient_json_info['bloodpressure'])],
                    'Skinthickness': [float(patient_json_info['skinthickness'])],
                    'Insulin': [float(patient_json_info['insulin'])],
                    'Bmi': [float(patient_json_info['bmi'])],
                    'Diabetespedigreefunction': [float(patient_json_info['diabetespedigreefunction'])],
                    'Age': [float(patient_json_info['age'])],
                    'Outcome': [1],
                }


                df = pd.DataFrame(data)
                path_to_dataset = os.path.join(settings.BASE_DIR, 'prepare_ml_model/data/')
                df.to_csv(path_to_dataset + 'diabetes.csv', mode='a', index=False, header=False)

                print("Data appended successfully.")

                context = {
                    'info': 'success',
                    'message': 'Data appended successfully',
                }
                return render(request, 'result.html', context)


        except ValueError as ve:
            context = {
                'info': 'success',
                'message': 'Could Not Save Data',
            }
            return render(request, 'result.html', context)