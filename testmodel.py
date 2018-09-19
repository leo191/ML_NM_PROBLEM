import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import base

IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"


training_set = base.load_csv_with_header(filename=IRIS_TRAINING,target_dtype=np.int32, features_dtype=np.float32)
test_set = base.load_csv_with_header(filename=IRIS_TEST,target_dtype=np.int32, features_dtype=np.float32)

#training_set = tf.data.Dataset.from
print(training_set)
print(test_set)


