from django import forms

class PatientForm(forms.Form):
    class Meta:
              fields = "__all__"

    pregnancies = forms.IntegerField()
    glucose = forms.IntegerField()
    bloodpressure = forms.IntegerField()
    skinthickness = forms.IntegerField()
    insulin = forms.IntegerField()
    bmi = forms.FloatField()
    diabetespedigreefunction = forms.FloatField()
    age = forms.IntegerField()

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