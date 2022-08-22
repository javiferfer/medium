""" Test models """

import json
import sys
import requests

from sklearn.metrics import accuracy_score

sys.path.append('.')
import src.utils.mnist_reader as mnist_reader


# Import the test data and split the data from the labels
X_test, y_test = mnist_reader.load_mnist('data/external/fashion', kind='t10k')

print(f'Shape X test: {X_test.shape}')
print(f'Shape y test: {y_test.shape}')

# Build the inference request
inference_request = {
    "inputs": [
        {
          "name": "predict",
          "shape": X_test.shape,
          "datatype": "FP64",
          "data": X_test.tolist()
        }
    ]
}

def infer(model_name, version):
    """" Send the prediction request to the relevant model """

    endpoint = f"http://localhost:8080/v2/models/{model_name}/versions/{version}/infer"
    response = requests.post(endpoint, json=inference_request)

    # Calculate accuracy
    y_pred = json.loads(response.text)['outputs'][0]['data']
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy for {model_name}: {accuracy}')

infer("fashion-xgboost", "v1")
infer("fashion-sklearn", "v1")
