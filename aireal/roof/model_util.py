import tensorflow as tf


def pool_ma(input_layer, pool_size, conv_size, strides, kernel_size):
    mp = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding="same")(input_layer)
    ap = tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=strides, padding="same")(input_layer)
    c = tf.keras.layers.Concatenate(axis=-1)([mp, ap])
    return tf.keras.layers.Conv2D(
        conv_size,
        kernel_size=kernel_size,
        activation='elu',
        padding='same',
        kernel_initializer='he_normal')(c)


def down_output(input_layer, size):
    return tf.keras.layers.Conv2D(
        size, kernel_size=(1, 1), activation='elu', padding='same',
        kernel_initializer='he_normal')(input_layer)

