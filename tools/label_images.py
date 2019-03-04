import tensorflow as tf
import util
import os
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib


from PIL import Image
import sys
import json


t = 'beach'
l = []
for path, subdirs, files in os.walk(f'/home/gsvigruha/images/landscape/water/{t}'):
  for name in files:
    d = os.path.join(path, name)
    l.append(d)
l.sort()


f=None
i = 500

if os.path.isfile(f'{t}.json'): 
  with open(f'{t}.json', 'r') as jf:
    final_tags = json.load(jf)
else:
  final_tags={}


def printimage():
  img=Image.open(l[i])
  plt.imshow(img)
  plt.draw()
  print(l[i])


def set_tag(i, t):
  if l[i] not in final_tags:
    final_tags[l[i]] = []

  if t in final_tags[l[i]]:
    final_tags[l[i]].remove(t)
  else:
    final_tags[l[i]].append(t)


def press(event):
    global i
    sys.stdout.flush()
    if event.key == 'right':
      with open(f'{t}.json', 'w') as outfile:
        json.dump(final_tags, outfile)
      i = i + 1
      if i % 10 == 0:
        recreate()
      else:
        if i >= 0 and i < len(l):
          printimage()
    elif event.key == 'left':
      i = i - 1
      if i >= 0 and i < len(l):
        printimage()
    else:
      if event.key == '1':
        set_tag(i, 'desert')
        print('desert')
      elif event.key == '2':
        set_tag(i, 'forest')
        print('forest')
      elif event.key == '3':
        set_tag(i, 'meadow')
        print('meadow')
      elif event.key == '4':
        set_tag(i, 'mountain')
        print('mountain')
      elif event.key == '5':
        set_tag(i, 'urban')
        print('urban')
      elif event.key == '6':
        set_tag(i, 'water')
        print('water')


def recreate():
  global f
  if f is not None:
    plt.close(f)
  f = plt.figure(figsize=(10, 10))
  f.canvas.mpl_disconnect(f.canvas.manager.key_press_handler_id)
  f.canvas.mpl_connect('key_press_event', press)
  printimage()
  plt.show()

recreate()

