import tensorflow as tf
import numpy as np
import pandas as pd
import math
import preprocessing
from efficientnet import efficientnet_builder
import os
tf.enable_eager_execution()

class DataLoader(object):
  def __init__(self, image_size):
    self.path = path = os.path.join(os.getcwd(), "train_dir")
    df = pd.read_csv('train.csv')
    # id_code,diagnosis
    self.names = np.asarray(df.id_code,dtype=str)
    self.labels = np.asarray(df.diagnosis, dtype=int)
    self.image_size = image_size
    # manual train test split
    train_size = int(math.ceil(self.names.shape[0] * 0.8))
    x_train = self.names[:train_size]
    y_train = self.labels[:train_size]
    x_test = self.names[train_size:]
    y_test = self.labels[train_size:]
    self.train_dataset = tf.compat.v1.data.Dataset.from_tensor_slices((x_train, y_train))
    self.test_dataset = tf.compat.v1.data.Dataset.from_tensor_slices((x_test, y_test))
  
  def _parse_func(self, is_training=True):
    filename = names+".png"
    filename = os.path.join(self.path, filename)
    read = tf.io.read_file(filename)
    decode = tf.image.decode_image(read)
    image = tf.cast(decode, tf.float32)
    # TODO: RANDOM CROP, MEAN

  def get_test_image(self):
    test_file = os.path.join(self.path, self.names[0]+".png")
    if tf.io.gfile.exists(test_file):
      read = tf.io.read_file(test_file)
      print(tf.image.decode_image(read).shape)

def main():
  model_name = 'efficientnet-b5' 
  _, _, image_size, _ = efficientnet_builder.efficientnet_params(model_name)
  # efficientnet-b5 should be 465x465
  loader = DataLoader(image_size)
  loader.get_test_image()

if __name__=="__main__":
  main()
