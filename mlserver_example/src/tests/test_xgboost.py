""" Test XGBoost """

import sys
import requests

sys.path.append('.')
import src.utils.mnist_reader as mnist_reader


ENDPOINT = "http://localhost:8080/v2/models/fashion-xgboost/versions/v1/infer"


#Import test data, grab the first row and corresponding label
# Load the test data
X_test, y_test = mnist_reader.load_mnist('data/external/fashion', kind='t10k')

#Prediction request parameters
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

#Make request and print response
response = requests.post(ENDPOINT, json=inference_request)
print(y_test)
