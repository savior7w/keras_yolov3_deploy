import requests
import os
import argparse

# Initialize the PyTorch REST API endpoint URL.
PyTorch_REST_API_URL = 'http://192.168.60.92:5000/predict'


def predict_result(image_path):
    # Initialize image path
    image = open(image_path, 'rb').read()
    payload = {'image': image}

    # Submit the request.
    r = requests.post(PyTorch_REST_API_URL, files=payload).json()

    # Ensure the request was successful.
    if r['success']:
        print(r['predictions'])



if __name__ == '__main__':
    path_file='./image_db/'
    total_name = os.listdir(path_file)
    print(total_name)
    for i in total_name:
        print(path_file+i)
        predict_result(path_file+i)

