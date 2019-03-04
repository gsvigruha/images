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
        files.append((d + '/' + f, d[d.rfind('/')+1:]))
  for d in ds:
    _load_dir(d)
  random.shuffle(files)
  train_files = files[0:3200]
  test_files = files[3200:4000]

  train_x = [f[0] for f in train_files]
  train_y = [label_map[f[1]] for f in train_files]
  test_x  = [f[0] for f in test_files]
  test_y  = [label_map[f[1]] for f in test_files]
  print(len(train_y), len(test_y))
  return train_x, train_y, test_x, test_y


train_x, train_y, test_x, test_y = _load_dirs([
  '/home/gsvigruha/images/lighting/night',
  '/home/gsvigruha/images/lighting/sunset',
  '/home/gsvigruha/images/lighting/rainy',
  '/home/gsvigruha/images/lighting/day'
])



def _vector(img, label):
  img = img / 255.0
  return {'input_images': img}, label



train_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(train_x), tf.constant(train_y)))
train_dataset = train_dataset.map(_parse_function).map(_vector).cache().batch(100).repeat()


input_f = tf.keras.layers.Input(shape=(64,), name='input_features')
input_i = tf.keras.layers.Input(shape=(256,256,3,), name='input_images')


conv_11 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                 activation='relu', padding='same',
                 kernel_initializer='he_normal')(input_i)
pool_11 = tf.keras.layers.AveragePooling2D((2, 2))(conv_11)

do_0 = tf.keras.layers.Dropout(0.2)(pool_11)

conv_12 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                 activation='relu', padding='same',
                 kernel_initializer='he_normal')(do_0)
pool_13 = tf.keras.layers.MaxPooling2D((2, 2))(conv_12)

do_1 = tf.keras.layers.Dropout(0.2)(pool_13)

conv_22 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                 activation='relu', padding='same',
                 kernel_initializer='he_normal')(do_1)
pool_23 = tf.keras.layers.AveragePooling2D((4, 4))(conv_22)


do_2 = tf.keras.layers.Dropout(0.25)(pool_23)


conv_3 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                 activation='relu', padding='same',
                 kernel_initializer='he_normal')(do_2)
pool_4 = tf.keras.layers.AveragePooling2D((4, 4))(conv_3)

do_2 = tf.keras.layers.Dropout(0.5)(pool_4)

conv_7 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                 activation='relu', padding='same',
                 kernel_initializer='he_normal')(do_2)
pool_8 = tf.keras.layers.MaxPooling2D((4, 4))(conv_7)

do_83 = tf.keras.layers.Dropout(0.5)(pool_8)

flatten_1 = tf.keras.layers.Flatten()(do_83)

out = tf.keras.layers.Dense(4, activation='softmax')(flatten_1)
model1 = tf.keras.models.Model(inputs=[input_i], outputs=out)


model1.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model1.summary()

test_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(test_x), tf.constant(test_y)))
test_dataset = test_dataset.map(_parse_function).map(_vector).cache().batch(40).repeat()

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50),
             tf.keras.callbacks.ModelCheckpoint(filepath='lighting.c3.best.h5', monitor='val_loss', save_best_only=True)]

model1.fit(train_dataset, epochs=200, steps_per_epoch=32, validation_data=test_dataset, validation_steps=20, callbacks=callbacks)


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
 
tf.keras.models.save_model(model1, '/home/gsvigruha/lighting.model.c3.h5', include_optimizer=False)
