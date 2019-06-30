import os
os.environ['CUDA_VISIBLE_DEVICE'] = '3'
import tensorflow as tf
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from efficientnet import efficientnet_builder
import matplotlib.pyplot as plt
from utils_ops import Conv2D_Pad
from PIL import Image
import time

MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]

class DataLoader(object):
  def __init__(self, image_size):
    self.path = path = os.path.join(os.getcwd(), "train_dir")
    df = pd.read_csv('train.csv')
    # id_code,diagnosis
    self.names = np.asarray(df.id_code, dtype=str)
    def pad_dir_name(x):
      filename = x+".png"
      filename = os.path.join(self.path, filename)
      return filename
    self.names = np.array([pad_dir_name(name) for name in self.names])
    self.labels = np.asarray(df.diagnosis, dtype=int)
    self.image_size = image_size
    # manual train test split
    train_size = int(math.ceil(self.names.shape[0] * 0.8))
    self.x_train = self.names[:train_size]
    self.y_train = self.labels[:train_size]
    self.x_val = self.names[train_size:]
    self.y_val = self.labels[train_size:]
    assert self.x_train.shape[0] == self.y_train.shape[0]
    assert self.x_val.shape[0] == self.y_val.shape[0]
    self.train_dataset = tf.compat.v1.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
    self.val_dataset = tf.compat.v1.data.Dataset.from_tensor_slices((self.x_val, self.y_val))
    self.is_cuda = tf.test.is_built_with_cuda() and tf.test.is_gpu_available()
    self.ckpt_dir = os.path.join(os.getcwd(), "ckpt_dir")
    self.model_dir = os.path.join(os.getcwd(), "model_dir")
    self.batch_size=4
    self.num_batch_per_epoch = (train_size + self.batch_size - 1) // self.batch_size

  def build_datasets(self):
    self._build_train_dataset()
    self._build_val_dataset()

  def _build_train_dataset(self):
    self.train_dataset = self.train_dataset.map(
      self._parse_func_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    self.train_dataset = self.train_dataset.shuffle(buffer_size=8)
    self.train_dataset = self.train_dataset.repeat()
    self.train_dataset = self.train_dataset.batch(self.batch_size)
    self.train_dataset = self.train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    self.train_iterator = tf.compat.v1.data.make_one_shot_iterator(self.train_dataset)

  def _build_val_dataset(self):
    self.val_dataset = self.val_dataset.map(
      self._parse_func_val, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    self.val_dataset = self.val_dataset.shuffle(buffer_size=8)
    self.val_dataset = self.val_dataset.repeat()
    self.val_dataset = self.val_dataset.batch(self.batch_size)
    self.val_iterator = tf.compat.v1.data.make_one_shot_iterator(self.val_dataset)

  def _transform(self, name, is_training=False):
    read = tf.io.read_file(name)
    img = tf.image.decode_png(read, channels=3)
    img = tf.cast(img, tf.float32)
    img /= 255.0
    img -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=img.dtype)
    img /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=img.dtype)
    img = tf.compat.v1.image.resize_image_with_pad(
      img, self.image_size, self.image_size,
      method=tf.image.ResizeMethod.BICUBIC)
    if is_training:
      log_probs = tf.math.log([[10., 10., 10., 10.]])
      k = tf.random.categorical(log_probs, 1, dtype=tf.int32)
      img = tf.image.random_flip_left_right(img)
      img = tf.image.random_flip_up_down(img)
      img = tf.image.rot90(img, k[0][0])
      img = tf.image.random_brightness(img, 0.1)
      img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
      img = tf.image.random_hue(img, max_delta=0.1)
      img = tf.image.random_contrast(img, 0.7, 1.3)
    return img

  def _parse_func_train(self, names, labels):
    img = self._transform(names, is_training=True)
    if self.is_cuda:
      img = tf.transpose(img, [2,0,1])
      img = tf.reshape(img, [3, self.image_size, self.image_size], name='reshape_img_first')
    else:
      img = tf.reshape(img, [self.image_size, self.image_size, 3], name='reshape_img_last')
    return (img, labels)
  
  def _parse_func_val(self, names, labels):
    img = self._transform(names)
    if self.is_cuda:
      img = tf.transpose(img, [2,0,1])
      img = tf.reshape(img, [3, self.image_size, self.image_size])
    else:
      img = tf.reshape(img, [self.image_size, self.image_size, 3])
    return (img, labels)
    
  def get_test_image(self, i):
    test_file = self.get_image_path(i)
    if tf.io.gfile.exists(test_file):
      img = self._transform(test_file)
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

def restore_model(sess, ckpt_dir, enable_ema=True, export_ckpt=None):
  sess.run(tf.compat.v1.global_variables_initializer())
  ckpt_dir = os.path.join(ckpt_dir, 'efficient')
  ckpt = tf.compat.v1.train.latest_checkpoint(ckpt_dir)
  if enable_ema:
    ema = tf.compat.v1.train.ExponentialMovingAverage(decay=0.0)
    ema_vars = (tf.compat.v1.trainable_variables() +
     tf.compat.v1.moving_average_variables())
    for v in tf.compat.v1.global_variables():
      if ('moving_mean' in v.name or 
          'moving_variance' in v.name):
        ema_vars.append(v)
    ema_vars = list(set(ema_vars))
    var_dict = ema.variables_to_restore(ema_vars)
    ema_assign_op = ema.apply(ema_vars)
  else:
    var_dict = None
    ema_assign_op = None

  sess.run(tf.compat.v1.global_variables_initializer())
  saver = tf.compat.v1.train.Saver(var_dict, max_to_keep=1)
  saver.restore(sess, ckpt)

  if export_ckpt:
    if ema_assign_op is not None:
      sess.run(ema_assign_op)
    saver = tf.compat.v1.train.Saver(max_to_keep=1)
    saver.save(sess, export_ckpt)

class FinalClassify(tf.keras.layers.Layer):
  def __init__(self, bn_axis, data_format, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
    super(FinalClassify, self).__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
    self.conv1 = Conv2D_Pad(256, 3, 1, data_format=data_format)
    self.bn1 = tf.keras.layers.BatchNormalization(axis=bn_axis)
    self.relu = tf.keras.layers.ReLU()
    self.conv2 = Conv2D_Pad(128, 1, 1, data_format=data_format)
    self.bn2 = tf.keras.layers.BatchNormalization(axis=bn_axis)
    self.avg = tf.keras.layers.GlobalAveragePooling2D(data_format=data_format)
    self.dropout = tf.keras.layers.Dropout(0.25)
    self.linear = tf.keras.layers.Dense(5, activation=None)

  def call(self, inputs, training=True):
    x = self.conv1(inputs)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)
    x = self.avg(x)
    x = self.dropout(x)
    x = self.linear(x)
    return x

def main():
  model_name = 'efficientnet-b5' 
  _, _, image_size, _ = efficientnet_builder.efficientnet_params(model_name)

  g = tf.Graph()
  with tf.compat.v1.Session(graph=g) as sess:
    loader = DataLoader(image_size)
    # loader.save_sample_images(10)
    loader.build_datasets()
    data_format = 'channels_first' if loader.is_cuda else 'channels_last'
    bn_axis = 1 if loader.is_cuda else -1
    batch_x, batch_y = loader.train_iterator.get_next()
    logits, _, model = efficientnet_builder.build_model_base(
      batch_x, model_name, training=False, data_format=data_format)
    
    restore_model(sess, loader.ckpt_dir)
    
    # NOTE: transfer learning
    model.trainable = False

    global_step = tf.Variable(
      0, dtype=tf.int32, trainable=False, name="global_step")

    with tf.compat.v1.variable_scope("final"):
      final_classifier = FinalClassify(bn_axis, data_format)
      logits = final_classifier(logits)

    log_probs = tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=batch_y
    )

    loss = tf.reduce_mean(log_probs)
    preds = tf.argmax(logits, axis=1)
    
    train_acc = tf.equal(preds, batch_y)
    train_acc = tf.cast(train_acc, tf.int32)
    train_acc = tf.reduce_sum(train_acc)

    # TODO: LR schedule rate.
    # lr = tf.compat.v1.train.exponential_decay(
    #   0.03, tf.maximum((global_step - 50), 0),
    #   100, 0.05, staircase=True
    # )
    # lr = tf.maximum(lr, 0.0001)
    final_var = [var for var in tf.compat.v1.trainable_variables() if var.name.startswith('final')]
    saver = tf.compat.v1.train.Saver(final_var, max_to_keep=1)

    train_op = (tf.compat.v1.train.GradientDescentOptimizer(0.001, use_locking=True)
            .minimize(loss, 
              global_step=global_step, 
              var_list=final_var))

    sess.run(tf.compat.v1.global_variables_initializer())

    # train phrase
    start_time = time.time()
    for i in range(loader.num_batch_per_epoch * 10):
      _, acc, loss_val = sess.run([train_op, train_acc, loss])
      if i % 50 == 0:
        ran = float(start_time - time.time() )/ 60.0
        tf.compat.v1.logging.info("Min: %.2f, Step: %d, train_acc: %d, loss_val: %d" % (ran, i, acc, loss_val))
      
      if i % loader.num_batch_per_epoch == 0:
        final_dir = os.path.join(loader.ckpt_dir, 'final')
        if not os.path.exists(final_dir):
          os.makedirs(final_dir)
        saver.save(sess, final_dir, global_step=global_step)

    # TODO: set to testphrase keras.


if __name__=="__main__":
  main()
