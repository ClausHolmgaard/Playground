import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from helpers import *

DATA_DIR = r'~/data/mnist_kaggle'
COMPETITION = r'digit-recognizer'
PREDICTION_FILE = r'./results/res.csv'

class model_logreg(object):
  def __init__(self):
    self.sess = None

  def fit(self, X_train, y_train, X_test, y_test, steps=1000):
    # Ground truth
    self.y_ = tf.placeholder(tf.float32, [None, 10])
    # Input
    self.x = tf.placeholder(tf.float32, [None, 784])
    
    # Weights
    self.W = tf.Variable(tf.zeros([784, 10]))
    #bias
    self.b = tf.Variable(tf.zeros(10))

    self.y = tf.matmul(self.x, self.W) + self.b

    self.predict = tf.argmax(self.y, axis=1)
    self.correct_prediction = tf.equal(self.predict, tf.argmax(self.y_, 1))
    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))

    #train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
    self.train_step = tf.train.AdamOptimizer(0.1).minimize(self.cost)

    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())

    for i in range(1, steps):
      cost, acc = self.sess.run([self.train_step, self.accuracy],
                                feed_dict={self.x:X_train, self.y_:y_train})
      if(i%10 == 0):
        print("Run {}, accuracy {}".format(i, acc))

    print("")
    train_acc = self.sess.run(self.accuracy, feed_dict={self.x:X_train, self.y_:y_train})
    print("Train accuracy: {}".format(train_acc))
    if X_test is not None and y_test is not None:
      test_acc = self.sess.run(self.accuracy, feed_dict={self.x:X_test, self.y_:y_test})
      print("Test accuracy: {}".format(test_acc))

  def do_predict(self, X, out_file):
    with open(out_file, 'w') as f:
      f.write("ImageId,Label\n")
      for counter, x in enumerate(X):
        pred = self.sess.run(self.predict, feed_dict={self.x:[x]})
        f.write("{},{}\n".format(counter+1, pred[0]))

if __name__ == "__main__":
  data_dir = get_data_dir(DATA_DIR)
  download_kaggle_dataset(COMPETITION, data_dir)

  train_file = os.path.join(data_dir, "train.csv")
  sub_file = os.path.join(data_dir, "test.csv")

  X_train, X_test, y_train, y_test = load_data(train_file, split=True)
  X_sub = load_sub_data(sub_file)

  print("X_train: {}".format(X_train.shape))
  print("X_test: {}".format(X_test.shape))
  print("y_train: {}".format(y_train.shape))
  print("y_test: {}".format(y_test.shape))

  print("")
  print("Running model, starting timer...")
  st = time.time()
  model = model_logreg()
  model.fit(X_train, y_train, X_test, y_test, steps=1000)
  run_time = time.time() - st
  print("")
  print("All done.")
  if run_time < 360:
    print("Run time {} secs".format(run_time))
  else:
    print("Run time {} mins".format(run_time/60))
  
  print("")
  print("Doing predictions...")
  model.do_predict(X_sub, PREDICTION_FILE)
  print("Done.")
