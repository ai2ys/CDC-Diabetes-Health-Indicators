import pickle

from flask import Flask
from flask import request
from flask import jsonify


def load(filename: str):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)


dv = load('dv.bin')
model = load('model.bin')

app = Flask('pre-diabetic-classifier')


def preprocess(patient: dict):
    pass
    # TODO: do preprocessing here
    patient_preproc = patient
    return patient_preproc

@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()
    patient = preprocess(patient)

    X = dv.transform([patient])
    y_pred = model.predict_proba(X)[0, 1]
    get_diabetic = y_pred >= 0.5

    result = {
        'get_diabetic_probability': float(y_pred),
        'get_diabetic': bool(get_diabetic)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)