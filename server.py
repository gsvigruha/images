from http.server  import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import threading
from concurrent.futures import ThreadPoolExecutor # pip install futures
from urllib.parse import urlparse, parse_qs
import urllib.request
import tempfile
import tensorflow as tf
import numpy as np
import time
import json

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

size =256


# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string, channels=3)
  image_resized = tf.image.resize_images(image_decoded, [size, size], method=tf.image.ResizeMethod.BILINEAR)
  img = tf.to_float(image_resized)
  img = img / 255.0
  return tf.stack([img], axis=0)

sess = tf.Session('', tf.Graph())

models = {}
with sess.graph.as_default():
    models['forest']   = {'model': tf.keras.models.load_model('models/forest.model.c3.h5')}
    models['meadow']   = {'model': tf.keras.models.load_model('models/meadow.model.c3.h5')}
    models['mountain'] = {'model': tf.keras.models.load_model('models/mountain.model.c3.h5')}
    models['urban']    = {'model': tf.keras.models.load_model('models/urban.model.c3.h5')}
    models['water']    = {'model': tf.keras.models.load_model('models/water.model.c3.h5')}
    models['desert']   = {'model': tf.keras.models.load_model('models/desert.model.c3.h5')}
    models['lighting'] = {
        'model': tf.keras.models.load_model('models/lighting.model.c3.h5'),
        'labels': ['night', 'sunset', 'rainy', 'day']
    }


class Handler(BaseHTTPRequestHandler):
    
    def do_GET(self):
        parsed_url = urlparse(self.path)
        if parsed_url.path == '/score':
            parsed_params = parse_qs(parsed_url.query)
            if parsed_params:
                url = parsed_params['url'][0]
                requested_models = [model_type for model_param in parsed_params['models'] for model_type in model_param.split(',')]
                try:
                    img_response = urllib.request.urlopen(url).read()
                except Exception as e:
                    print(str(e))
                    self.send_response(500)
                    self.end_headers()
                    msg = f'Error happened while downloading url: {e}.'
                    self.wfile.write(msg.encode('utf-8'))
                    return
                with tempfile.NamedTemporaryFile() as f:
                    f.write(img_response)
                    f.flush()
                    f.seek(0)
                    with sess.graph.as_default():
                        img = _parse_function(f.name)
                        img_array = sess.run(img)
                        predictions = {}
                        for model in requested_models:
                            if 'labels' in models[model]:
                                label_predictions = {}
                                scores = models[model]['model'].predict(img_array)[0]
                                for i in range(0, len(models[model]['labels'])):
                                    label = models[model]['labels'][i]
                                    label_predictions[label] = float(scores[i])
                                    predictions[model] = label_predictions
                            else:
                                score = float(models[model]['model'].predict(img_array)[0][0])
                                predictions[model] = score
                        result = {
                            'url': url,
                            'predictions': predictions
                        }
                        self.send_response(200)
                        self.end_headers()
                        self.wfile.write(json.dumps(result).encode('utf-8'))
                        return
        else:
            if parsed_url.path == '/' or parsed_url.path == '':
                path = '/index.html'
            else:
                path = parsed_url.path
            try:
                with open("web/app" + path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self.send_response(404)
                self.end_headers()                



class PoolMixIn(ThreadingMixIn):
    def process_request(self, request, client_address):
        self.pool.submit(self.process_request_thread, request, client_address)


class PoolHTTPServer(PoolMixIn, HTTPServer):
    pool = ThreadPoolExecutor(max_workers=10)


if __name__ == '__main__':
    server = PoolHTTPServer(('', 8080), Handler)
    print('Starting server, use <Ctrl-C> to stop')
    server.serve_forever()


#http://localhost:8080/score?url=https%3A%2F%2Fdaily.jstor.org%2Fwp-content%2Fuploads%2F2016%2F10%2FMoving_Forest_1050_700.jpg&model=forest
