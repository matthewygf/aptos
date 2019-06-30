import tensorflow as tf
from tensorflow.keras import layers, Model

class Conv2D_Pad(layers.Layer):
  def __init__(self, planes, kernel_size, strides, data_format='channels_first', use_bias=False, name=None):
    super(Conv2D_Pad, self).__init__()
    self.data_format = data_format
    self.use_bias = use_bias
    self.planes = planes
    self.kernel_size = kernel_size
    self.strides = strides
    self.conv2d = layers.Conv2D(planes, kernel_size, strides=strides, 
                                padding=('same' if strides==1 else 'valid'), use_bias=use_bias,
                                kernel_initializer=tf.compat.v1.keras.initializers.he_normal(),
                                data_format=self.data_format,
                                name=name)
  
  def call(self, inputs):
    x = inputs
    
    if self.strides > 1:
      x = self.fixed_padding(x)
    
    x = self.conv2d(x)
    return x
  
  def fixed_padding(self, inputs):
    pad_total = self.kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if self.data_format == 'channels_first':
      padded_inputs = tf.pad(tensor=inputs,
                           paddings=[[0, 0], [0, 0], [pad_beg, pad_end],
                                     [pad_beg, pad_end]])
    else:
      padded_inputs = tf.pad(tensor=inputs,
                            paddings=[[0, 0], [pad_beg, pad_end],
                                      [pad_beg, pad_end], [0, 0]])
    return padded_inputs
