import tensorflow as tf
import numpy as np
import math

size = 256
ranges = [2, 4, 8, 16, 32, 64, 128, 256]
#ranges = [1, 3, 9, 27, 81, 256]

def _extract_freq(img, range_from, range_to, parts=1):
  r,g,b = tf.split(img, 3, 2)
  r = tf.squeeze(r)
  g = tf.squeeze(g)
  b = tf.squeeze(b)
  k = 1.0 - tf.math.maximum(tf.math.maximum(r, g), b)
  y = (1.0 - b - k) / tf.maximum(1.0 - k, 1e-6)
  p = (1.0 - g - k) / tf.maximum(1.0 - k, 1e-6)
  c = (1.0 - r - k) / tf.maximum(1.0 - k, 1e-6)
  w = (r + b + g) / 3.0
  
  const = tf.constant([float(x + 1) for x in range(0, 256)])
  r_f = tf.math.abs(tf.spectral.dct(r)[:,range_from:range_to]) * const[range_from:range_to] / 256.0
  g_f = tf.math.abs(tf.spectral.dct(g)[:,range_from:range_to]) * const[range_from:range_to] / 256.0
  b_f = tf.math.abs(tf.spectral.dct(b)[:,range_from:range_to]) * const[range_from:range_to] / 256.0
  w_f = tf.math.abs(tf.spectral.dct(w)[:,range_from:range_to]) * const[range_from:range_to] / 256.0
  y_f = tf.math.abs(tf.spectral.dct(y)[:,range_from:range_to]) * const[range_from:range_to] / 256.0
  p_f = tf.math.abs(tf.spectral.dct(p)[:,range_from:range_to]) * const[range_from:range_to] / 256.0
  c_f = tf.math.abs(tf.spectral.dct(c)[:,range_from:range_to]) * const[range_from:range_to] / 256.0
  result = []
  interval = math.ceil(256.0 / parts)
  for p in range(0, parts):
    l = p * interval
    h = p * interval + interval
    result.extend([
      tf.math.reduce_mean(r_f[l:h]),
      tf.math.reduce_mean(g_f[l:h]),
      tf.math.reduce_mean(b_f[l:h]),
      tf.math.reduce_mean(w_f[l:h])
    #  tf.math.reduce_mean(y_f[l:h])
    #  tf.math.reduce_mean(p_f[l:h]),
    #  tf.math.reduce_mean(c_f[l:h])
    ])
  return result


def freq_hist(img, py=1, px=1, ranges=ranges):
  img = img / 255.0
  result = []
  r_l = 0
  for r_u in ranges:
    result_y = _extract_freq(img, r_l, r_u, py)
    result.extend(result_y)
    result_x = _extract_freq(tf.image.transpose_image(img), r_l, r_u, px)
    result.extend(result_x)
    r_l = r_u

  return tf.stack(result)


def color_hist(img):
  img = img / 255.0
  bin_size = 0.25
  hist_entries = []

  r, g, b = tf.split(img, 3, 2)
  for idx_r, i_r in enumerate(np.arange(0, 1, bin_size)):
    for idx_g, i_g in enumerate(np.arange(0, 1, bin_size)):
      for idx_b, i_b in enumerate(np.arange(0, 1, bin_size)):
        gt_r = tf.greater(r, i_r)
        leq_r = tf.less_equal(r, i_r + bin_size)
        gt_g = tf.greater(g, i_g)
        leq_g = tf.less_equal(g, i_g + bin_size)
        gt_b = tf.greater(b, i_b)
        leq_b = tf.less_equal(b, i_b + bin_size)
        logical_fn = tf.logical_and(gt_r, tf.logical_and(leq_r, tf.logical_and(gt_g, tf.logical_and(leq_r, tf.logical_and(gt_b, leq_b)))))
        hist_entries.append(tf.reduce_sum(tf.cast(logical_fn, tf.float32))) 

  tf_v = tf.stack(hist_entries)
  return tf_v / tf.maximum(tf.reduce_sum(tf_v), tf.constant(1.0))


def hsv_hist(img):
  img = tf.image.rgb_to_hsv(img)
  h, s, v = tf.split(img, 3, 2)
  h_bins = 24
  s_bins = 12
  v_bins = 12
  h_h = tf.math.bincount(tf.dtypes.cast(tf.math.ceil(h * h_bins),  tf.int32), minlength=h_bins+1, maxlength=h_bins+1)[1:]
  s_h = tf.math.bincount(tf.dtypes.cast(tf.math.floor(s * s_bins), tf.int32), minlength=s_bins+1, maxlength=s_bins+1)[1:]
  v_h = tf.math.bincount(tf.dtypes.cast(tf.math.floor(v * v_bins), tf.int32), minlength=v_bins+1, maxlength=v_bins+1)[1:]
  h_h = tf.to_float(h_h / tf.maximum(tf.reduce_sum(h_h), 1))
  v_h = tf.to_float(v_h / tf.maximum(tf.reduce_sum(v_h), 1))
  s_h = tf.to_float(s_h / tf.maximum(tf.reduce_sum(s_h), 1))
  return tf.concat([h_h, v_h, s_h], axis=0)


def bb(mask):
  indices_y = tf.expand_dims(tf.range(256, delta=1, dtype=tf.int32), 1)
  masked_indices_y_low = tf.math.add((mask - 1.0) * -256, tf.to_float(indices_y))
  masked_indices_y_high = tf.math.add((mask - 1.0) * 256, tf.to_float(indices_y))

  indices_x = tf.transpose(indices_y)
  masked_indices_x_low = tf.math.add((mask - 1.0) * -256, tf.to_float(indices_x))
  masked_indices_x_high = tf.math.add((mask - 1.0) * 256, tf.to_float(indices_x))
  return tf.reduce_min(masked_indices_x_low), tf.reduce_max(masked_indices_x_high), tf.reduce_min(masked_indices_y_low), tf.reduce_max(masked_indices_y_high)


def comps(img, threshold=0.2):
  with tf.device('/cpu:0'):
    img = img / 255.0
    img_stack = tf.stack([img])
    di = tf.image.sobel_edges(img_stack)
    dy, dx = tf.split(di, 2, axis=4)
    dy = tf.reduce_sum(tf.math.abs(dy), axis=3) / 3.0
    dx = tf.reduce_sum(tf.math.abs(dx), axis=3) / 3.0
    edges = tf.math.maximum(tf.squeeze(dy), tf.squeeze(dx))

    flat = tf.less(edges, threshold)  
    comps = tf.contrib.image.connected_components(flat)
    c = tf.math.bincount(tf.reshape(comps, [256 * 256]), minlength=6)[1:]
    cnt = tf.math.top_k(c, 5)

    r = []
    for i in range(0, 4):
      flat_mask = tf.to_float(tf.equal(tf.fill([256, 256], tf.squeeze(cnt.indices[i]) + 1), comps))
      mask = tf.expand_dims(flat_mask, 2)
      meta = tf.squeeze(tf.stack(list(bb(flat_mask)) + [tf.to_float(cnt.values[i])], axis=0))
      r.append((tf.math.multiply(img, mask), meta))

  return r


def sky(img, thresholds=[0.2, 0.33, 0.5]):
  c = []
  for t in thresholds:
    c.extend(comps(img, t))
  metas = tf.stack([x[1] for x in c], axis=0)
  zeros = tf.stack([0.0 for _ in c], axis=0)
  sizes = tf.squeeze(metas[:,4:5])
  min_y = tf.squeeze(metas[:,2:3])
  cond = tf.logical_and(
    tf.equal(min_y, zeros),
    tf.logical_and(
      tf.greater(sizes, tf.stack([256 * 256 / 16 for _ in c], axis=0)),
      tf.less(sizes, tf.stack([256 * 256 / 4 for _ in c], axis=0))
  ))
  filtered = tf.where(cond, sizes, zeros)
  cnt = tf.math.top_k(filtered, 1)
  return tf.stack([x[0] for x in c])[cnt.indices[0]]
  

def regions(img):
  by_y = tf.stack(tf.split(img, 4, 0), 0)
  by_x = tf.stack(tf.split(by_y, 4, 2), 0)
  return tf.reshape(by_x, [16, 64, 64, 3])

