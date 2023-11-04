import requests
import json

patient = json.load(open('test_sample.json', 'r'))
print(json.dumps(patient, indent=4))

url = 'http://localhost:9696/predict'
response = requests.post(url, json=patient)
print(response)
result = response.json() 
print(result)