""" Test xgboost """

import sys
import xgboost as xgb

from sklearn.metrics import accuracy_score

sys.path.append('.')
import src.utils.mnist_reader as mnist_reader

FILENAME = 'src/models/xgboost/fashion_mnist_xgboost.json'


# Import test data, grab the first row and corresponding label
X_test, y_test = mnist_reader.load_mnist('data/external/fashion', kind='t10k')
dtest = xgb.DMatrix(X_test, label=y_test)

# Load the weights
model = xgb.Booster()
model.load_model(FILENAME)

# Inferences
y_pred = model.predict(dtest)
print(accuracy_score(y_test, y_pred))
