import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from matplotlib.pyplot import figure
import tensorflow as tf

from images.aireal.roof.image_loader import LABELS, TRAIN, BATCH_SIZE



def feature_iter_1(test_file_list, model_file):
  sess = tf.Session('', tf.Graph())
  with sess.graph.as_default():
    train_images=[]
    model = tf.keras.models.load_model(model_file)
    for filename in test_file_list:
      rgb_image = tf.read_file(TRAIN + filename + ".jpg")
      rgb_image_decoded = tf.image.decode_jpeg(rgb_image, channels=3)
      rgb_image_decoded = tf.to_float(rgb_image_decoded) / 255.0

      cir_fn = tf.strings.regex_replace(filename, '2005050310033_78642723578549', '2005050310034_78642723578549_CIR')
      cir_image = tf.read_file(TRAIN + cir_fn + ".jpg")
      cir_image_decoded = tf.image.decode_jpeg(cir_image, channels=3)
      cir_image_decoded = tf.to_float(cir_image_decoded) / 255.0
      train_image_decoded = tf.concat([rgb_image_decoded, cir_image_decoded], axis=2)
      input_tf = sess.run(tf.stack([train_image_decoded], axis=0))
      train_images.append(model.predict(input_tf))
  sess.close()
  return np.squeeze(np.stack(train_images, axis=0))


def feature_iter_2(test_file_list, model_file, output_dirs):
  sess = tf.Session('', tf.Graph())
  with sess.graph.as_default():
    train_images=[]
    model = tf.keras.models.load_model(model_file)
    for filename in test_file_list:
      rgb_image = tf.read_file(TRAIN + filename + ".jpg")
      rgb_image_decoded = tf.image.decode_jpeg(rgb_image, channels=3)
      rgb_image_decoded = tf.to_float(rgb_image_decoded) / 255.0

      cir_fn = tf.strings.regex_replace(filename, '2005050310033_78642723578549', '2005050310034_78642723578549_CIR')
      cir_image = tf.read_file(TRAIN + cir_fn + ".jpg")
      cir_image_decoded = tf.image.decode_jpeg(cir_image, channels=3)
      cir_image_decoded = tf.to_float(cir_image_decoded) / 255.0

      feature_tensors = [rgb_image_decoded, cir_image_decoded]
      for output_dir in output_dirs:
        prev_image = tf.read_file(TRAIN + output_dir + filename + "_roof_output.png")
        prev_image_decoded = tf.image.decode_jpeg(prev_image, channels=1)
        prev_image_decoded = tf.to_float(prev_image_decoded) / 255.0
        feature_tensors.append(prev_image_decoded)
    
      train_image_decoded = tf.concat(feature_tensors, axis=2)

      input_tf = sess.run(tf.stack([train_image_decoded], axis=0))
      train_images.append(model.predict(input_tf))
  sess.close()
  return np.squeeze(np.stack(train_images, axis=0))


def show(test_file_list, y):

  f = figure(num=None, figsize=(16, 32), dpi=80, facecolor='w', edgecolor='k')
  N = len(test_file_list)

  for i in range(0, N):
    name = test_file_list[i]
    print(name)
    f.add_subplot(N,3,i*3+1)
    plt.imshow(np.squeeze(y[i]), cmap='gray', vmin=0, vmax=1)
    img=mpimg.imread('/home/gsvigruha/aireal/Classification/'+name+'.jpg')
    f.add_subplot(N,3,i*3+2)
    plt.imshow(img)
    img_s=mpimg.imread('/home/gsvigruha/aireal/Classification/'+name+'_shapes.png')
    f.add_subplot(N,3,i*3+3)
    plt.imshow(img_s)

  plt.show()
