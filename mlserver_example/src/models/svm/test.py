""" Test sklearn """

import sys
import pickle
import joblib

from sklearn import svm

sys.path.append('.')
import src.utils.mnist_reader as mnist_reader


FILENAME = 'src/models/svm/fashion_mnist_svm.joblib'


# Import test data, grab the first row and corresponding label
X_test, y_test = mnist_reader.load_mnist('data/external/fashion', kind='t10k')

# Load the weights
model = joblib.load(FILENAME)

# Inferences
result = model.score(X_test, y_test)
print(result)
