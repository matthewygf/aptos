import os
import tensorflow as tf
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from efficientnet import efficientnet_builder
import matplotlib.pyplot as plt
from utils_ops import Conv2D_Pad
from sklearn.model_selection import train_test_split
from PIL import Image
import time

MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]

FEATURES_ONLY = False
FINE_TUNE = True

class DataLoader(object):
  def __init__(self, image_size):
    self.path = path = os.path.join(os.getcwd(), "train_dir")
    # id_code,diagnosis
    df = pd.read_csv('train.csv')
    # Balance the distribution in the training set
    train_ids, val_ids = train_test_split(df.id_code, test_size=0.25)
    train_df = df[df.id_code.isin(train_ids)]
    val_df = df[df.id_code.isin(val_ids)]
    new_train_df = train_df.groupby(['diagnosis']).apply(
      lambda x: x.sample(450, replace=True)).reset_index(drop=True)
    
    def pad_dir_name(x):
      filename = x+".png"
      filename = os.path.join(self.path, filename)
      return filename

    new_train_df['id_code'] = new_train_df['id_code'].apply(
      lambda x: pad_dir_name(x))
    new_train_df = new_train_df.sample(frac=1).reset_index(drop=True)
    print(new_train_df.tail())
    print(new_train_df.groupby(['diagnosis']).count())
    time.sleep(2)
    val_df['id_code'] = val_df['id_code'].apply(
      lambda x: pad_dir_name(x))

    self.image_size = image_size
    # manual train test split
    train_size = new_train_df.shape[0]
    self.x_train = np.asarray(new_train_df.id_code, dtype=str)
    self.y_train = np.asarray(new_train_df.diagnosis, dtype=int)
    self.x_val = np.asarray(val_df.id_code, dtype=str)
    val_size = self.x_val.shape[0]
    self.y_val = np.asarray(val_df.diagnosis, dtype=int)
    assert self.x_train.shape[0] == self.y_train.shape[0]
    assert self.x_val.shape[0] == self.y_val.shape[0]
    self.train_dataset = tf.compat.v1.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
    self.val_dataset = tf.compat.v1.data.Dataset.from_tensor_slices((self.x_val, self.y_val))
    self.is_cuda = tf.test.is_built_with_cuda() and tf.test.is_gpu_available()
    self.ckpt_dir = os.path.join(os.getcwd(), "ckpt_dir")
    self.model_dir = os.path.join(os.getcwd(), "model_dir")
    self.batch_size=16
    self.num_batch_per_epoch = (train_size + self.batch_size - 1) // self.batch_size
    self.num_val_batch = (val_size + self.batch_size - 1) // self.batch_size

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

    # crop
    should_crop = tf.math.log([[70., 30.]])
    crop = tf.random.categorical(should_crop, 1, dtype=tf.int32)
    crop_and_resize = tf.image.crop_and_resize(
      [img], boxes=[[0.1, 0.1, 0.9, 0.9]], 
      crop_size=[self.image_size, self.image_size],
      box_ind=[0])
    resize_only = tf.compat.v1.image.resize_image_with_pad(
      img, self.image_size, self.image_size,
      method=tf.image.ResizeMethod.BICUBIC)
    img = tf.cond(
      tf.equal(crop[0][0], 1),
      lambda: tf.identity(crop_and_resize),
      lambda: tf.identity(resize_only)
    )
    should_augment = tf.math.log([[80., 20.]])
    aug = tf.random.categorical(should_augment, 1, dtype=tf.int32)

    def transform_further(img):
      img = tf.image.random_brightness(img, 0.2)
      cond = tf.random.uniform([], maxval=1, dtype=tf.int32)

      def saturate_hue_contrast(img):
        img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
        img = tf.image.random_hue(img, max_delta=0.1)
        img = tf.image.random_contrast(img, 0.7, 1.3)
        return img

      return tf.cond(tf.equal(cond, 1),
          lambda: saturate_hue_contrast(img),
          lambda: tf.identity(img)
        )

    if is_training:
      log_probs = tf.math.log([[10., 10., 10., 10.]])
      k = tf.random.categorical(log_probs, 1, dtype=tf.int32)
      img = tf.image.random_flip_left_right(img)
      img = tf.image.random_flip_up_down(img)
      img = tf.image.rot90(img, k[0][0])
      
    return tf.cond(tf.equal(aug[0][0], 1),
          lambda: transform_further(img),
          lambda: tf.identity(img))

  def _parse_func_train(self, names, labels):
    img = self._transform(names, is_training=True)
    if self.is_cuda:
      img = tf.transpose(img, [0,3,1,2])
      img = tf.reshape(img, [3, self.image_size, self.image_size], name='reshape_img_first')
    else:
      img = tf.reshape(img, [self.image_size, self.image_size, 3], name='reshape_img_last')
    return (img, labels)
  
  def _parse_func_val(self, names, labels):
    img = self._transform(names)
    if self.is_cuda:
      img = tf.transpose(img, [0,3,1,2])
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

def restore_efficient(sess, ckpt_dir, enable_ema=True, export_ckpt=None):
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
    ema_vars_new = [var for var in ema_vars if 'efficient' in var.name]
    ema_vars_new = list(set(ema_vars_new))
    # NOTE: only restore efficient
    var_dict = ema.variables_to_restore(ema_vars_new)
    ema_assign_op = ema.apply(ema_vars_new)
  else:
    var_dict = [var for var in tf.compat.v1.trainable_variables() if 'efficient' in var.name]
    ema_assign_op = None

  sess.run(tf.compat.v1.global_variables_initializer())
  restore_vars = {}
  for k,v in var_dict.items():
    if 'efficient' in k:
      restore_vars[k] = v
  saver = tf.compat.v1.train.Saver(restore_vars, max_to_keep=1)
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
    self.dropout = tf.keras.layers.Dropout(0.2)
    self.fc1 = tf.keras.layers.Dense(2048, activation='relu')
    self.linear = tf.keras.layers.Dense(5, activation=None)

  def call(self, inputs, training=True):
    x = self.conv1(inputs)
    x = self.bn1(x, training=training)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.bn2(x, training=training)
    x = self.relu(x)
    x = self.avg(x)
    x = self.dropout(x, training=training)
    x = self.fc1(x)
    x = self.linear(x)
    return x

class Driver(object):
  def __init__(self, model_name, data_format, bn_axis):
    self.model_name = model_name
    self.data_format = data_format
    self.bn_axis = bn_axis
    self.model_name = model_name

  def build_graph(self, features, training=False, reuse=False):
    if training:
      print('------- building training graph --------')
    else:
      print('------- building test graph --------')

    logits, endpoints, model = efficientnet_builder.build_model_base(
      features, self.model_name, 
      reuse=reuse, training=training, 
      data_format=self.data_format,
      feature_only=FEATURES_ONLY)

    if not FINE_TUNE:
      # NOTE: transfer learning
      model.trainable = False
    else:
      model.trainable = True

    #print(endpoints)

    if FEATURES_ONLY:
      with tf.compat.v1.variable_scope("final", reuse=reuse):
        final_classifier = FinalClassify(self.bn_axis, self.data_format)
        logits = final_classifier(logits, training=training)
    else:
      logits = tf.keras.layers.Dense(5)(tf.keras.layers.ReLU()(logits))
      
    return logits, model
  
  def build_train_ops(self, global_step, features, labels):
    self.train_logits, train_m = self.build_graph(features, training=True)
    self.train_log = tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(
      logits=self.train_logits, labels=labels)

    self.loss = tf.reduce_mean(self.train_log)
    preds = tf.argmax(self.train_logits, axis=1)
    
    self.train_acc = tf.equal(preds, labels)
    self.train_acc = tf.cast(self.train_acc, tf.int32)
    self.train_acc = tf.reduce_sum(self.train_acc)
    if FEATURES_ONLY:
      self.final_var = [var for var in tf.compat.v1.trainable_variables() 
        if 'final' in var.name]
    else:
      self.final_var = tf.compat.v1.trainable_variables()

    # : LR schedule rate.
    lr = tf.compat.v1.train.exponential_decay(
      0.03, tf.maximum((global_step - 50), 0),
      100, 0.05, staircase=True
    )
    lr = tf.maximum(lr, 0.0001)
    
    self.train_op = (tf.compat.v1.train.GradientDescentOptimizer(lr, use_locking=True)
            .minimize(self.loss, 
              global_step=global_step, 
              var_list=self.final_var))

  def build_val_ops(self, features, labels):
    self.val_logits, val_m = self.build_graph(features, training=False, reuse=True)
    preds = tf.argmax(self.val_logits, axis=1)
    self.val_acc = tf.equal(preds, labels)
    self.val_acc = tf.cast(self.val_acc, tf.int32)
    self.val_acc = tf.reduce_sum(self.val_acc)


def main():
  model_name = 'efficientnet-b3' 
  _, _, image_size, _ = efficientnet_builder.efficientnet_params(model_name)

  g = tf.Graph()
  with tf.compat.v1.Session(graph=g) as sess:
    loader = DataLoader(image_size)
    # loader.save_sample_images(10)
    loader.build_datasets()
    data_format = 'channels_first' if loader.is_cuda else 'channels_last'
    bn_axis = 1 if loader.is_cuda else -1
    batch_x, batch_y = loader.train_iterator.get_next()
    val_batch_x, val_batch_y = loader.val_iterator.get_next()

    global_step = tf.Variable(
      0, dtype=tf.int32, trainable=False, name="global_step")

    driver = Driver(model_name, data_format, bn_axis)

    driver.build_train_ops(global_step, batch_x, batch_y)
    driver.build_val_ops(val_batch_x, val_batch_y)

    if FEATURES_ONLY:
      restore_efficient(sess, loader.ckpt_dir)

    saver = tf.compat.v1.train.Saver(driver.final_var, max_to_keep=1)

    sess.run(tf.compat.v1.global_variables_initializer())

    # train phrase
    start_time = time.time()
    for i in range(loader.num_batch_per_epoch * 300):
      _, acc, loss_val = sess.run([driver.train_op, driver.train_acc, driver.loss])
      if i % 50 == 0:
        ran = float(time.time() - start_time  )/ 60.0
        acc = float(acc/loader.batch_size)
        print("Min: %.2f, Step: %d, train_acc: %.3f, loss_val: %.4f" % (ran, i, acc, loss_val))
      
      if i % loader.num_batch_per_epoch == 0:
        val_acc = 0.0
        for j in range(loader.num_val_batch):
          val_acc += float(sess.run(driver.val_acc) / loader.batch_size)
        val_acc /= loader.num_val_batch
        epoch = int(i // loader.num_batch_per_epoch)
        print("At epoch %d, valid accuracy %.3f" %(epoch, val_acc))

        final_dir = os.path.join(loader.ckpt_dir, 'final')
        model_name = os.path.join(final_dir, 'model_classify')
        if not os.path.exists(final_dir):
          os.makedirs(final_dir)
        saver.save(sess, model_name, global_step=global_step)

    # TODO: set to testphrase keras.


if __name__=="__main__":
  main()
