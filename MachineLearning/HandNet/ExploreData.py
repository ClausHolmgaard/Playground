import os
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = r'/home/claus/data/HandNet/TrainData'


if __name__ == "__main__":
  print("Data dir: " + DATA_PATH)
  for f in os.listdir(DATA_PATH):
    print(f)

