""" Train sklearn """

import time
import sys
import joblib

from sklearn import svm

sys.path.append('.')
import src.utils.mnist_reader as mnist_reader


# Load training data
X_train, y_train = mnist_reader.load_mnist('data/external/fashion', kind='train')
model = svm.SVC(kernel="poly", degree=4, gamma=0.1)

# Train Model
start = time.time()
model.fit(X_train, y_train)
end = time.time()
exec_time = end - start
print(f'Execution time: {exec_time} seconds')

# Save Model
joblib.dump(model, "src/models/svm/fashion_mnist_svm.joblib")
