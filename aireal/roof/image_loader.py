from functools import partial
from PIL import Image
import numpy as np
import tensorflow as tf


LABELS = '/home/gsvigruha/aireal/Labelled/Roof/'
TRAIN = '/home/gsvigruha/aireal/Classification/'


BATCH_SIZE = 3


def load_image_input_features_and_labels(filename):
  rgb_image = tf.read_file(TRAIN + filename + ".jpg")
  rgb_image_decoded = tf.image.decode_jpeg(rgb_image, channels=3)
  rgb_image_decoded = tf.to_float(rgb_image_decoded) / 255.0

  cir_fn = tf.strings.regex_replace(filename, '2005050310033_78642723578549', '2005050310034_78642723578549_CIR')
  cir_image = tf.read_file(TRAIN + cir_fn + ".jpg")
  cir_image_decoded = tf.image.decode_jpeg(cir_image, channels=3)
  cir_image_decoded = tf.to_float(cir_image_decoded) / 255.0
  train_image_decoded = tf.concat([rgb_image_decoded, cir_image_decoded], axis=2)

  test_image = tf.read_file(LABELS + filename + "_shapes.png")
  image_decoded = tf.image.decode_png(test_image, channels=3)
  r, g, b = tf.split(image_decoded, 3, 2)
  test_image_decoded = tf.to_float(r) / 255.0
  return train_image_decoded, test_image_decoded



def save_image_output_features_and_labels(train_file_list, test_file_list, model_file, output_dir):
  sess = tf.Session('', tf.Graph())

  with sess.graph.as_default():
    prev_model = tf.keras.models.load_model(model_file)
    for filename in train_file_list + test_file_list:
      rgb_image = tf.read_file(TRAIN + filename + ".jpg")
      rgb_image_decoded = tf.image.decode_jpeg(rgb_image, channels=3)
      rgb_image_decoded = tf.to_float(rgb_image_decoded) / 255.0

      cir_fn = tf.strings.regex_replace(filename, '2005050310033_78642723578549', '2005050310034_78642723578549_CIR')
      cir_image = tf.read_file(TRAIN + cir_fn + ".jpg")
      cir_image_decoded = tf.image.decode_jpeg(cir_image, channels=3)
      cir_image_decoded = tf.to_float(cir_image_decoded) / 255.0
      train_image_decoded = tf.concat([rgb_image_decoded, cir_image_decoded], axis=2)
      input_tf = sess.run(tf.stack([train_image_decoded], axis=0))
      prev_output = prev_model.predict(input_tf)
      nparr = np.uint8(np.squeeze(prev_output) * 255)
      Image.fromarray(nparr, 'L').save(output_dir + filename + "_roof_output.png")
  sess.close()



def _load_image_input_iterated_features_and_labels(filename, output_dirs: list):
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
    
    test_image = tf.read_file(LABELS + filename + "_shapes.png")
    image_decoded = tf.image.decode_png(test_image, channels=3)
    r, g, b = tf.split(image_decoded, 3, 2)
    test_image_decoded = tf.to_float(r) / 255.0
    return train_image_decoded, test_image_decoded

def load_image_input_iterated_features_and_labels(output_dirs):
    return partial(_load_image_input_iterated_features_and_labels, output_dirs=output_dirs)


def create_datasets(test_file_list, train_file_list, load_image_input_features_and_labels_fn):
  test_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(test_file_list))
  test_dataset = test_dataset.map(load_image_input_features_and_labels_fn).cache().batch(BATCH_SIZE).repeat()

  train_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(train_file_list))
  train_dataset = train_dataset.map(load_image_input_features_and_labels_fn).shuffle(200).cache().batch(BATCH_SIZE).repeat()

  return train_dataset, test_dataset

