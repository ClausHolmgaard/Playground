import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from helpers import *

DATA_DIR = r'~/data/mnist_kaggle'
COMPETITION = r'digit-recognizer'
PREDICTION_FILE = r'./results/res.csv'

class NNModel(object):
  def __init__(self):
    self.sess = None
  
  def do_fit(self, X_train, y_train, X_test, y_test, steps=1000, lr=0.01, decay=False):
    self.x = tf.placeholder(tf.float32, [None, 784])
    self.y_ = tf.placeholder(tf.float32, [None, 10])

    initializer = tf.contrib.layers.xavier_initializer()

    layer1 = tf.layers.dense(inputs=self.x, units=2048, activation=tf.nn.leaky_relu, kernel_initializer=initializer)
    layer2 = tf.layers.dense(inputs=layer1, units=2048, activation=tf.nn.leaky_relu, kernel_initializer=initializer)
    layer3 = tf.layers.dense(inputs=layer2, units=2048, activation=tf.nn.leaky_relu, kernel_initializer=initializer)
    layer4 = tf.layers.dense(inputs=layer3, units=2048, activation=tf.nn.leaky_relu, kernel_initializer=initializer)
    layer5 = tf.layers.dense(inputs=layer4, units=512, activation=tf.nn.leaky_relu, kernel_initializer=initializer)
    layer6 = tf.layers.dense(inputs=layer5, units=256, activation=tf.nn.leaky_relu, kernel_initializer=initializer)
    self.y = tf.layers.dense(inputs=layer6, units=10, activation=tf.nn.sigmoid, kernel_initializer=initializer)

    self.predict = tf.argmax(self.y, axis=1)
    self.correct_prediction = tf.equal(self.predict, tf.argmax(self.y_, 1))
    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
  
    self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))

    if decay:
      self.global_step = tf.Variable(0, trainable=False)
      self.learning_rate = tf.train.exponential_decay(lr,
                                                self.global_step,
                                                100,
                                                0.9,
                                                staircase=False)
      self.train_step = tf.train.AdamOptimizer(lr).minimize(self.cost, global_step=self.global_step)
    else:
      self.learning_rate = tf.constant(lr)
      self.train_step = tf.train.AdamOptimizer(lr).minimize(self.cost)

    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())

    print("")
    for i in range(1, steps):
      cost, acc, lrate = self.sess.run([self.train_step, self.accuracy, self.learning_rate],
                                feed_dict={self.x:X_train, self.y_:y_train})
              
      if(i%10 == 0):
        print("Run {}, accuracy {} with learning rate {}".format(i, acc, lrate))

    print("")
    train_acc = self.sess.run(self.accuracy, feed_dict={self.x:X_train, self.y_:y_train})
    print("Train accuracy: {}".format(train_acc))
    if X_test is not None and y_test is not None:
      test_acc = self.sess.run(self.accuracy, feed_dict={self.x:X_test, self.y_:y_test})
      print("Test accuracy: {}".format(test_acc))

  def do_predict(self, X, out_file):
    pass

if __name__ == "__main__":
  data_dir = get_data_dir(DATA_DIR)
  #download_kaggle_dataset(COMPETITION, data_dir)

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
  model = NNModel()
  model.do_fit(X_train, y_train, X_test, y_test, steps=2000, lr=1e-3, decay=True)
  run_time = time.time() - st
  print("")
  print("All done.")
  if run_time < 360:
    print("Run time {} secs".format(run_time))
  else:
    print("Run time {} mins".format(run_time/60))
  