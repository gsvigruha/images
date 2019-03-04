import tensorflow as tf
import util
import os
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

size = 256


# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string, channels=3)
  image_resized = tf.image.resize_images(image_decoded, [size, size], method=tf.image.ResizeMethod.BICUBIC)
  image_resized = tf.to_float(image_resized) / 255.0
  r,g,b = tf.split(image_resized, 3, 2)
  b = tf.squeeze(b)
  g = tf.squeeze(g)
  r = tf.squeeze(r)
  return tf.stack([r,g,b], axis=2)


img = _parse_function('file:///home/gsvigruha/images/landscape/water/river_lake/img_000583.jpg')


with tf.Session() as sess:
  img_plt = sess.run(img)
  f = plt.figure(figsize=(20, 15))
  f.add_subplot(1,4,1)
  plt.imshow(img_plt)
  f.add_subplot(1,4,2)
  plt.imshow(sess.run(util.regions(img, 'x'))[0])
  f.add_subplot(1,4,3)
  plt.imshow(sess.run(util.regions(img, 'x'))[1])
  f.add_subplot(1,4,4)
  plt.imshow(sess.run(util.regions(img, 'x'))[2])
  
  plt.show(block=True)
