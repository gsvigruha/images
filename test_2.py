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
  image_resized = tf.image.resize_images(image_decoded, [size, size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return tf.to_float(image_resized)


img = _parse_function('/home/gsvigruha/images/lighting/day/img_000729.jpg')


comps = util.comps(img, threshold=0.2) + util.comps(img, threshold=0.33) + util.comps(img, threshold=0.5) + util.comps(img, threshold=0.66)


with tf.Session() as sess:
  img_plt = sess.run(img / 255.0)
  f = plt.figure(figsize=(20, 15))
  f.add_subplot(2, 1 + (2 + len(comps)) / 2, 1)
  plt.imshow(img_plt)
  j = 1
  
  for comp in comps:
    #print(sess.run(util.hsv_hist(comp)))
    m_plt = sess.run(comp)
    print(m_plt[1])
    f.add_subplot(2, 1 + (2 + len(comps)) / 2, 1 + j)
    plt.imshow(m_plt[0])
    j = j + 1
  
  sky = util.sky(img)
  f.add_subplot(2, 1 + (2 + len(comps)) / 2, 2 + j)
  plt.imshow(sess.run(sky))
  
  plt.show(block=True)
