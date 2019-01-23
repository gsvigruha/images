import os

d='/home/gsvigruha/images/lighting/day'
l=os.listdir(d)
i = 0
for f in l:
  ext = f[f.rfind('.'):]
  os.rename(d + '/' + f, d + '/img_' + str(i).rjust(6, '0') + ext)
  i = i + 1

