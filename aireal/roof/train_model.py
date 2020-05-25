import tensorflow as tf
import random
import os
import math


from images.aireal.roof.image_loader import LABELS, TRAIN, BATCH_SIZE, \
  create_datasets, load_image_input_features_and_labels, save_image_output_features_and_labels, load_image_input_iterated_features_and_labels, load_image_input_iterated_features_and_labels
from images.aireal.roof import model_architecture_v1, model_architecture_v2

SEED = 12345

file_list = []
for path, subdirs, files in os.walk(LABELS):
  for name in files:
    d = os.path.join(path, name)
    if name.lower().endswith('_shapes.png'):
      file_list.append(name[:-11])

random.Random(SEED).shuffle(file_list)

num_files = len(file_list)
num_train = 56 # int(num_file * 0.8)
num_test = num_files - num_train

train_file_list= file_list[:num_train]
test_file_list= file_list[num_train:]
print(train_file_list)
print(test_file_list)


train_dataset, test_dataset = create_datasets(test_file_list, train_file_list, load_image_input_features_and_labels)

model_1 = model_architecture_v1.create_model_v1()
callbacks_1 = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50),
             tf.keras.callbacks.ModelCheckpoint(filepath='roof.best.iter1.h5', monitor='val_loss', save_best_only=True)]
model_1.fit(train_dataset, epochs=50, steps_per_epoch=num_train/BATCH_SIZE, validation_data=test_dataset,
           validation_steps=num_test/BATCH_SIZE, class_weight={0:1.0, 1:1.0}, callbacks=callbacks_1)

save_image_output_features_and_labels(train_file_list, test_file_list, model_file='roof.best.iter1.h5', output_dir=TRAIN+'output_1/')

model_2 = model_architecture_v1.create_model_v1()
callbacks_2 = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50),
             tf.keras.callbacks.ModelCheckpoint(filepath='roof.best.iter2.h5', monitor='val_loss', save_best_only=True)]
model_2.fit(train_dataset, epochs=50, steps_per_epoch=num_train/BATCH_SIZE, validation_data=test_dataset,
           validation_steps=num_test/BATCH_SIZE, class_weight={0:1.0, 1:1.0}, callbacks=callbacks_2)

save_image_output_features_and_labels(train_file_list, test_file_list, model_file='roof.best.iter2.h5', output_dir=TRAIN+'output_2/')


from images.aireal.roof import view_results
# view_results.show('roof.best.iter1.h5', test_file_list)
# view_results.show('roof.best.iter2.h5', test_file_list)

train_dataset_2, test_dataset_2 = create_datasets(test_file_list, train_file_list, load_image_input_iterated_features_and_labels(['output_1/', 'output_2/']))
model_combined = model_architecture_v2.create_model_v2(8)
callbacks_combined = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50),
             tf.keras.callbacks.ModelCheckpoint(filepath='roof.best.combined.h5', monitor='val_loss', save_best_only=True)]
model_combined.fit(train_dataset_2, epochs=50, steps_per_epoch=num_train/BATCH_SIZE, validation_data=test_dataset_2,
           validation_steps=num_test/BATCH_SIZE, class_weight={0:1.0, 1:1.0}, callbacks=callbacks_combined)

