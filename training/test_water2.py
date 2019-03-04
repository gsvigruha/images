import tensorflow as tf
import util
import os
import random
import numpy as np
import pandas as pd
import math
from functools import partial
import json

size = 256

tags = {}
with open('/home/gsvigruha/images/landscape/desert/tags.json') as tagsfile:
  tags.update(json.load(tagsfile))
with open('/home/gsvigruha/images/landscape/water/beach/tags.json') as tagsfile:
  tags.update(json.load(tagsfile))
with open('/home/gsvigruha/images/landscape/water/river_lake/tags.json') as tagsfile:
  tags.update(json.load(tagsfile))
with open('/home/gsvigruha/images/landscape/meadow/tags.json') as tagsfile:
  tags.update(json.load(tagsfile))
with open('/home/gsvigruha/images/landscape/forest/tags.json') as tagsfile:
  tags.update(json.load(tagsfile))
with open('/home/gsvigruha/images/landscape/mountain/tags.json') as tagsfile:
  tags.update(json.load(tagsfile))
with open('/home/gsvigruha/images/landscape/urban/city_european/tags.json') as tagsfile:
  tags.update(json.load(tagsfile))


# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label, sx, sy):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string, channels=3)
  image_resized = tf.image.resize_images(image_decoded, [sy, sx], method=tf.image.ResizeMethod.BILINEAR)
  return tf.to_float(image_resized), label



def _load_dirs():
  root = '/home/gsvigruha/images/landscape'
  positive_dir = 'water'
  excluded = 0

  negative_list = []
  positive_list = []
  for path, subdirs, files in os.walk(root):
    for name in files:
        d = os.path.join(path, name)
        if name.lower().endswith('jpg') or name.lower().endswith('jpeg'):
          if positive_dir in path:
            positive_list.append(d)
          else:
            if d not in tags or positive_dir not in tags[d]:
              negative_list.append(d)
            else:
              excluded = excluded + 1

  print(f'excluded: {excluded}')
  random.shuffle(negative_list)
  random.shuffle(positive_list)
  files = [(f, 0.0) for f in negative_list[:2000]] + [(f, 1.0) for f in positive_list[:2000]]
  random.shuffle(files)
  train_files = files[:3200]
  test_files = files[3200:]

  train_x = [f[0] for f in train_files]
  train_y = [f[1] for f in train_files]
  test_x  = [f[0] for f in test_files]
  test_y  = [f[1] for f in test_files]
  print(len(train_y), len(test_y))
  return train_x, train_y, test_x, test_y


train_x, train_y, test_x, test_y = _load_dirs()


def _vector(img, label):
  img = img / 255.0
  return {'input_images': images}, label


ensemble = False

train_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(train_x), tf.constant(train_y)))
train_dataset = train_dataset.map(partial(_parse_function, sx=256, sy=256)).map(_vector).cache().batch(50).repeat()


input_f = tf.keras.layers.Input(shape=(64,), name='input_features')
input_i = tf.keras.layers.Input(shape=(256,256,3,), name='input_images')


conv_1 = tf.keras.layers.Conv2D(64, kernel_size=(5, 5),
                 activation='relu', padding='same', strides=2,
                 kernel_initializer='he_normal')(input_i)
do_1 = tf.keras.layers.Dropout(0.4)(conv_1)
conv_2 = tf.keras.layers.Conv2D(48, kernel_size=(2, 2),
                 activation='relu', padding='same', strides=2,
                 kernel_initializer='he_normal')(do_1)
do_2 = tf.keras.layers.Dropout(0.3)(conv_2)
conv_3 = tf.keras.layers.Conv2D(32, kernel_size=(2, 2),
                 activation='relu', padding='same', strides=2,
                 kernel_initializer='he_normal')(do_2)
do_3 = tf.keras.layers.Dropout(0.25)(conv_3)
conv_4 = tf.keras.layers.Conv2D(24, kernel_size=(2, 2),
                 activation='relu', padding='same', strides=2,
                 kernel_initializer='he_normal')(conv_3)
do_4 = tf.keras.layers.Dropout(0.2)(conv_4)
conv_5 = tf.keras.layers.Conv2D(16, kernel_size=(2, 2),
                 activation='relu', padding='same', strides=2,
                 kernel_initializer='he_normal')(do_4)
do_5 = tf.keras.layers.Dropout(0.15)(conv_5)
conv_6 = tf.keras.layers.Conv2D(12, kernel_size=(2, 2),
                 activation='relu', padding='same', strides=2,
                 kernel_initializer='he_normal')(do_5)
do_6 = tf.keras.layers.Dropout(0.1)(conv_6)
conv_7 = tf.keras.layers.Conv2D(8, kernel_size=(2, 2),
                 activation='relu', padding='same', strides=2,
                 kernel_initializer='he_normal')(do_6)

do_0 = tf.keras.layers.Dropout(0.1)(conv_7)


flatten_1 = tf.keras.layers.Flatten()(do_0)

out = tf.keras.layers.Dense(1, activation='sigmoid')(flatten_1)
model1 = tf.keras.models.Model(inputs=[input_i], outputs=out)


model1.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model1.summary()

test_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(test_x), tf.constant(test_y)))
test_dataset = test_dataset.map(partial(_parse_function, sx=256, sy=256)).map(_vector).cache().batch(20).repeat()
model1.fit(train_dataset, epochs=100, steps_per_epoch=64, validation_data=test_dataset, validation_steps=40)


#loss, acc = model.evaluate(test_dataset.map(_vector).batch(40), steps=20)
#print(f'loss {loss}, acc {acc}')

test_steps = 40


y1 = model1.predict(test_dataset, steps=test_steps)


cm = [[0,0],[0,0]]
l_h = [0,0]
p_h = [0,0]
good = 0
good_1 = 0
good_sure_1 = 0
total_sure_1 = 0
good_sure_2 = 0
total_sure_2 = 0
for i in range(0, test_steps * 20):
  l = int(test_y[i])
  p1 = y1[i]
  pl1 = 1 if p1[0] > 0.5 else 0
  p = p1[0]
  pl = 1 if p > 0.5 else 0
  cm[pl][l] = cm[pl][l] + 1
  if l == pl:
    good = good + 1
  if l == pl1:
    good_1 = good_1 + 1
  if not l == pl:
    print(test_x[i], l, pl, p, pl1)
  l_h[l] = l_h[l] + 1
  p_h[pl] = p_h[pl] + 1
  if p >= 0.8 or p <= 0.2:
    if l == pl:
      good_sure_1 = good_sure_1 + 1
    total_sure_1 = total_sure_1 + 1
  if p >= 0.9 or p <= 0.1:
    if l == pl:
      good_sure_2 = good_sure_2 + 1
    total_sure_2 = total_sure_2 + 1


print(good / float(test_steps * 20))
print(good_1 / float(test_steps * 20))
print(f'{good_sure_1 / float(total_sure_1)} ({total_sure_1})')
print(f'{good_sure_2 / float(total_sure_2)} ({total_sure_2})')
print(l_h)
print(p_h)
print(pd.DataFrame(cm)) 
 
tf.keras.models.save_model(model1, '/home/gsvigruha/water2.model.c3.h5', include_optimizer=False)
