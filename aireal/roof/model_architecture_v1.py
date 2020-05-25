import tensorflow as tf

from images.aireal.roof.model_util import pool_ma


def create_model_v1():

  input_1 = tf.keras.layers.Input(shape=(875,875,6,))


  conv_1_1 = tf.keras.layers.Conv2D(12, kernel_size=(3, 3), activation='elu', padding='same',
                                 kernel_initializer='he_normal')(input_1)
  conv_1_2 = tf.keras.layers.Conv2D(12, kernel_size=(3, 3), activation='elu', padding='same',
                                 kernel_initializer='he_normal')(input_1)
  conv_1 = tf.keras.layers.Concatenate(axis=-1)([conv_1_1, conv_1_2])


  layer_5x5_shift = pool_ma(conv_1, (5, 5), 16, (1, 1), (3, 3))
  layer_5x5_tile = pool_ma(conv_1, (5, 5), 16, (5, 5), (3, 3))

  layer_25x25_shift = pool_ma(layer_5x5_tile, (5, 5), 12, (1, 1), (3, 3))
  layer_25x25_tile = pool_ma(layer_5x5_tile, (5, 5), 12, (5, 5), (3, 3))

  layer_125x125 = pool_ma(layer_25x25_tile, (5, 5), 8, (1, 1), (3, 3))
  layer_175x175 = pool_ma(layer_25x25_tile, (7, 7), 8, (1, 1), (3, 3))

  layer_125x125_conv = tf.keras.layers.Conv2D(4, kernel_size=(5, 5), activation='elu', padding='same',
                                 kernel_initializer='he_normal')(layer_25x25_tile)

  layer_25x25_conv = tf.keras.layers.Conv2D(8, kernel_size=(3, 3), activation='elu', padding='same',
                                 kernel_initializer='he_normal')(layer_25x25_tile)


  up_25x25 = tf.keras.layers.UpSampling2D(size=(5, 5), interpolation='nearest')(layer_25x25_shift)
  up_125x125 = tf.keras.layers.UpSampling2D(size=(25, 25), interpolation='nearest')(layer_125x125)
  up_175x175 = tf.keras.layers.UpSampling2D(size=(25, 25), interpolation='nearest')(layer_175x175)

  up_125x125_conv = tf.keras.layers.UpSampling2D(size=(25, 25), interpolation='nearest')(layer_125x125_conv)
  up_25x25_conv = tf.keras.layers.UpSampling2D(size=(25, 25), interpolation='nearest')(layer_25x25_conv)

  concat_2 = tf.keras.layers.Concatenate(axis=-1)([conv_1_2, layer_5x5_shift, up_25x25, up_125x125, up_175x175, up_25x25_conv])

  conv_f = tf.keras.layers.Conv2D(8, kernel_size=(1, 1),
                 activation='elu', padding='same',
                 kernel_initializer='he_normal')(concat_2)

  out = tf.keras.layers.Dense(1, activation='sigmoid')(conv_f)
  model = tf.keras.models.Model(inputs=[input_1], outputs=out)

  model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])

  model.summary()
  return model


