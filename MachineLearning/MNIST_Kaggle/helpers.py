import os
import subprocess
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def get_data_dir(path):
  if path.startswith('~'):
    home_dir = os.path.expanduser('~')
    data_dir = os.path.join(home_dir, path[2:])
    return data_dir
  else:
    return path

def download_kaggle_dataset(competition, path):
  print("Saving dataset to {}...".format(path))
  subprocess.run(["kaggle", "competitions", "download",
                  "-c", competition,
                  "-p", path])
  print("Download done.")

def load_data(csv_file, split=False, shape=None, test_size=0.20):
  dataset = pd.read_csv(csv_file)
  X = dataset.iloc[:, 1:].values
  y = dataset.iloc[:, 0:1].values

  # Normalize
  X = (X - np.min(X)) / (np.max(X) - np.min(X))

  # Reshape
  if shape is not None:
    X = X.reshape(X.shape[0], *shape, -1)

  # One hot labels
  ohe = OneHotEncoder()
  ohe.fit(y)
  y = ohe.transform(y).toarray()

  if split:
    return train_test_split(X, y, test_size=test_size)
  else:
    return X, y, None, None

def load_sub_data(csv_file, shape=None):
  dataset = pd.read_csv(csv_file)
  X = dataset.iloc[:].values

  # Normalize
  X = (X - np.min(X)) / (np.max(X) - np.min(X))

  # Reshape
  if shape is not None:
    X = X.reshape(X.shape[0], *shape, -1)
  
  return X

if __name__ == "__main__":
  load_sub_data(r'/home/claus/data/mnist_kaggle/test.csv')
