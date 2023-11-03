from django.shortcuts import render

# Create your views here.
import joblib
from rest_framework.response import Response
from django.conf import settings
import numpy as np
import os
from rest_framework.views import APIView
from .forms import PatientForm

# Load the model from the static folder
path_to_model = os.path.join(settings.BASE_DIR, 'static/model/')
loaded_model = joblib.load(open(path_to_model + 'diabete_detector_model.pkl', 'rb'))

# For API HTML View
class PatientStatusApiView(APIView):
    def get(self, request):
        return_data = {
            "error_code": "0",
            "info": "success",
            "extra": "it is a POST Api",
        }
        return Response(return_data)

    def post(self, request):
        try:
            form = PatientForm(request.POST)

            if form.is_valid():
                # load the request data
                patient_json_info = form.data

                # Retrieve all the values from the json data
                patient_info = np.array(list(patient_json_info.values()))

                # Make prediction
                patient_status = loaded_model.predict([patient_info])

                # Model confidence score
                model_confidence_score = np.max(loaded_model.predict_proba([patient_info]))

                model_prediction = {
                    'info': 'success',
                    'patient_status': patient_status[0],
                    'model_confidence_proba': float("{:.2f}".format(model_confidence_score * 100))
                }
            return Response(model_prediction)

        except ValueError as ve:
            model_prediction = {
                'error_code': '-1',
                "info": str(ve)
            }
        return Response(model_prediction)


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