import pickle
import json
import numpy as np
from flask import Flask
from flask import request
from flask import jsonify
from copy import deepcopy
from waitress import serve

def load(filename: str):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)


data_normalizers = json.load(open('data_normalizers.json', 'r'))
model = load('models/model_random_forest.bin')
dv = load('models/dv_random_forest.bin')
app = Flask('pre-diabetic-classifier')


def preprocess(
    patient: dict, 
    data_normalizers: dict) -> dict:
    patient_preprocessed = deepcopy(patient)

    # standard scaling (additional log for BMI)
    patient_preprocessed['BMI'] = np.log(patient_preprocessed['BMI'])
    patient_preprocessed['BMI'] -= data_normalizers['BMI']['mean']
    patient_preprocessed['BMI'] /= data_normalizers['BMI']['std']

    patient_preprocessed['Age'] -= data_normalizers['Age']['mean']
    patient_preprocessed['Age'] /= data_normalizers['Age']['std']
    
    # Min max scaling
    patient_preprocessed['MentHlth'] -= data_normalizers['MentHlth']['min']
    patient_preprocessed['MentHlth'] /= (data_normalizers['MentHlth']['max'] - data_normalizers['MentHlth']['min'])

    patient_preprocessed['PhysHlth'] -= data_normalizers['PhysHlth']['min']
    patient_preprocessed['PhysHlth'] /= (data_normalizers['PhysHlth']['max'] - data_normalizers['PhysHlth']['min'])

    patient_preprocessed['GenHlth'] -= data_normalizers['GenHlth']['min']
    patient_preprocessed['GenHlth'] /= (data_normalizers['GenHlth']['max'] - data_normalizers['GenHlth']['min'])

    patient_preprocessed['Education'] -= data_normalizers['Education']['min']
    patient_preprocessed['Education'] /= (data_normalizers['Education']['max'] - data_normalizers['Education']['min'])

    patient_preprocessed['Income'] -= data_normalizers['Income']['min']
    patient_preprocessed['Income'] /= (data_normalizers['Income']['max'] - data_normalizers['Income']['min'])
    #all other variables are binary, they do not need to be normalized

    y_true = None
    if 'Diabetes_binary' in patient_preprocessed.keys():
        y_true = patient_preprocessed['Diabetes_binary']
        del patient_preprocessed['Diabetes_binary']

    patient_preprocessed = dv.transform(patient_preprocessed)

    return patient_preprocessed, y_true

@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()
    patient_preprocessed = preprocess(patient, data_normalizers)

    X,_ = patient_preprocessed
    y_pred = model.predict_proba(X)[0, 1]
    get_diabetic = y_pred >= 0.5

    result = {
        'get_diabetic_probability': float(y_pred),
        'get_diabetic': bool(get_diabetic)
    }

    return jsonify(result)


if __name__ == "__main__":
    #app.run(debug=True, host='0.0.0.0', port=9696)
    serve(app, host='0.0.0.0', port=9696)