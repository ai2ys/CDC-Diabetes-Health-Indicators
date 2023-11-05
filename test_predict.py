import requests
import json
import argparse
import logging

def main(args):
    patient = json.load(open('test_sample.json', 'r'))
    print(json.dumps(patient, indent=4))

    url = args.url
    if not url.startswith('http://'):
        url = f'http://{url}'
    if not url.endswith('/'):
        url = f'{url}/'
    url = f'{url}predict'

    response = requests.post(url, json=patient)
    print(response)
    result = response.json() 
    print(result)


# Define the command line arguments
parser = argparse.ArgumentParser(description='Test the prediction service on a local machine or AWS Elastic Beanstalk')

parser.add_argument(
    '--url',
    # nargs='?',
    type=str, 
    default='http://localhost:9696/',
    help='The url to use for the prediction (default: "http://localhost:9696/").' \
        +' Alternatively, pass url to Elastic Beanstalk.', 
    )

if __name__ == "__main__":
    # Parse the command line arguments
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    main(args)