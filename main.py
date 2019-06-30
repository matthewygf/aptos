import tensorflow as tf
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from efficientnet import efficientnet_builder
import os
import matplotlib.pyplot as plt
from PIL import Image
tf.enable_eager_execution()

MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]

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
    image = tf.cast(decode, tf.float32) / 255
    # TODO: RANDOM CROP, MEAN
    

  def get_test_image(self, i):
    test_file = self.get_image_path(i)
    if tf.io.gfile.exists(test_file):
      read = tf.io.read_file(test_file)
      img = tf.image.decode_image(read, channels=3)
      img = tf.cast(img, tf.float32)
      img /= 255.0
      img -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=img.dtype)
      img /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=img.dtype)
      img = tf.compat.v1.image.resize_images(
        img, [self.image_size, self.image_size], 
        method=tf.image.ResizeMethod.BICUBIC,
        preserve_aspect_ratio=True)
      
      log_probs = tf.math.log([[10., 10., 10., 10.]])
      k = tf.random.categorical(log_probs, 1, dtype=tf.int32)
      img = tf.image.rot90(img, k[0][0])
      img = tf.image.random_flip_left_right(img)
      img = tf.image.random_flip_up_down(img)
      img = tf.image.random_brightness(img, 0.1)
      img = tf.image.random_contrast(img, 0.8, 1.2)
      return img

  def get_image_path(self,i):
    test_file = os.path.join(self.path, self.names[i]+".png")
    return test_file
  
  def save_sample_images(self, i):
    for index in range(i):
      img = self.get_test_image(index)
      img *= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=img.dtype)
      img += tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=img.dtype)
      img *= 255.0
      img = Image.fromarray(np.uint8(img.numpy()))
      img.save("test%d.png" % index)

def main():
  model_name = 'efficientnet-b5' 
  _, _, image_size, _ = efficientnet_builder.efficientnet_params(model_name)
  # efficientnet-b5 should be 465x465
  loader = DataLoader(image_size)
  loader.save_sample_images(10)
  # TODO: build model, fine tune

if __name__=="__main__":
  main()
