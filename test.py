import tensorflow as tf
import util
import os
import random
import numpy as np
import pandas as pd

size = 256


label_map = {'night': 0.0, 'sunset': 1.0, 'rainy': 2.0, 'day': 3.0}


# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string, channels=3)
  image_resized = tf.image.resize_images(image_decoded, [size, size])
  return image_resized, label


def _load_dirs(ds):
  files = []
  def _load_dir(d):
    for f in os.listdir(d):
      if (f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')):
        test = random.random() < 0.2
        files.append((d + '/' + f, d[d.rfind('/')+1:], test))
  for d in ds:
    _load_dir(d)
  random.shuffle(files)

  train_x = [f[0] for f in files if not f[2]]
  train_y = [label_map[f[1]] for f in files if not f[2]]
  test_x  = [f[0] for f in files if f[2]]
  test_y  = [label_map[f[1]] for f in files if f[2]]
  print(len(train_y), len(test_y))
  return train_x, train_y, test_x, test_y


train_x, train_y, test_x, test_y = _load_dirs([
  '/home/gsvigruha/images/lighting/night',
  '/home/gsvigruha/images/lighting/sunset',
  '/home/gsvigruha/images/lighting/rainy',
  '/home/gsvigruha/images/lighting/day'
])


def _vector(img, label):
  return tf.concat([
    util.color_hist(img[0:96,:,:]),
    util.color_hist(img[80:176,:,:]),
    util.color_hist(img[160:256,:,:]),
    util.freq_hist(img/ 255.0, 3, 1),
   # util.hsv_hist(img)
   # util.hsv_hist(img[0:96,:,:]),
   # util.hsv_hist(img[80:176,:,:]),
   # util.hsv_hist(img[160:256,:,:])
  ]# + [util.hsv_hist(c) for c in util.comps(img, threshold=0.2)] +
   #   [util.hsv_hist(c) for c in util.comps(img, threshold=0.33)] +
   #   [util.hsv_hist(c) for c in util.comps(img, threshold=0.5)]
  , 0), label


train_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(train_x), tf.constant(train_y)))
train_dataset = train_dataset.map(_parse_function).map(_vector).cache().batch(100).repeat()

dim = (64*3 + 
       8*6*1 + 8*6*3
       #48*3 
)
       #2*3*32)
print(dim)


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(dim,)),
    tf.keras.layers.Dense(64, activation='elu'),
    tf.keras.layers.Dense(4, activation='softmax')])

model.compile(optimizer=tf.train.AdamOptimizer(0.005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



model.fit(train_dataset, epochs=50, steps_per_epoch=32)

#loss, acc = model.evaluate(test_dataset.map(_vector).batch(40), steps=20)
#print(f'loss {loss}, acc {acc}')

test_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(test_x), tf.constant(test_y)))
test_dataset = test_dataset.map(_parse_function).map(_vector).batch(40)

test_steps = 20
y = model.predict(test_dataset, steps=test_steps)

cm = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
l_h = [0,0,0,0]
p_h = [0,0,0,0]
good = 0
good_sure = 0
total_sure = 0
for i in range(0, test_steps * 40):
  l = int(test_y[i])
  p = y[i]
  pl = int(np.argmax(p))
  cm[pl][l] = cm[pl][l] + 1
  if l == pl:
    good = good + 1
  else:
    print(test_x[i], l, pl, p)
  l_h[l] = l_h[l] + 1
  p_h[pl] = p_h[pl] + 1
  if np.amax(p) >= 0.7:
    if l == pl:
      good_sure = good_sure + 1
    total_sure = total_sure + 1

print(good / float(test_steps * 40))
print(f'{good_sure / float(total_sure)} ({total_sure})')
print(l_h)
print(p_h)
print(pd.DataFrame(cm)) 
 

