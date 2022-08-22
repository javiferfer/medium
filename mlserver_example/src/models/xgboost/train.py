""" Train XGBoost """

import time
import sys
import xgboost as xgb

sys.path.append('.')
import src.utils.mnist_reader as mnist_reader


NUM_ROUND = 50

# Load training data
X_train, y_train = mnist_reader.load_mnist('data/external/fashion', kind='train')
dtrain = xgb.DMatrix(X_train, label=y_train)

#Train Model
params = {
    'max_depth': 5,
    'eta': 0.3,
    'verbosity': 1,
    'objective': 'multi:softmax',
    'num_class' : 10
}

start = time.time()
bstmodel = xgb.train(params, dtrain, NUM_ROUND, evals=[(dtrain, 'label')], verbose_eval=10)
end = time.time()
exec_time = end - start
print(f'Execution time: {exec_time} seconds')

# Save model
bstmodel.save_model("src/models/xgboost/fashion_mnist_xgboost.json")
