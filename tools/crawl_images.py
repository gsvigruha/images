from html.parser import HTMLParser
from multiprocessing import Pool
import urllib.request
import re
import random
import logging


# create a subclass and override the handler methods
class MyHTMLParser(HTMLParser):
    
    links = set()

    def handle_starttag(self, tag, attrs):
      if tag == 'a':
        attrs = dict(attrs)
        if 'href' in attrs:
          url = attrs['href']
          self.links.add(url)

    def handle_endtag(self, tag):
      return

    def handle_data(self, data):
      return



def download_image(i):
    try:
      url = i[2:]
      url = url.replace('{width}', '256')
      fn = url.replace('/', '_')
      b = urllib.request.urlopen('http://'+url, timeout=2).read()
      if len(b) >= 50000:
        with open('/home/gsvigruha/images/tmp/' + fn, 'wb') as f:
          f.write(b)
          return True
    except Exception as e:
      print(e)
    return False

pool = Pool(30)


def process(url, processed, queue):
  if url in processed:
    return
  processed.add(url)
  parser = MyHTMLParser()
  with urllib.request.urlopen(url, timeout=2) as page:
    s = page.read().decode('utf8')
    parser.feed(s)
    images = re.findall(r'\/\/[\w\/.\-_\{\}]*\.jpg',s)
  
  print(f'Downloading {len(images)} images.')
  downloaded = sum(pool.map(download_image, images))
  print(f'Downloaded {downloaded} images.')

  candidates = []
  for link in parser.links:
    if not link.startswith('http://') and not link.startswith('https://'): 
        link = url + link
    if link not in processed:
      candidates.append(link)
  random.shuffle(candidates)
  queue.update(candidates[:5])
  

queue = set(['https://www.yahoo.com'])
processed = set()
for i in range(0, 1000):
  url = queue.pop()
  try:
    print('processing:' + url)
    process(url, processed, queue)
  except Exception as e:
    logging.error(e)


pool.join()
pool.close()
