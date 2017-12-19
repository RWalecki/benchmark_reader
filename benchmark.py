import glob 
import cv2
import time
import h5py
import shutil
import os
import random, string
import numpy as np

files = np.array(sorted(list(glob.glob('./data/*.bmp'))))
r = np.arange(len(files))
np.random.shuffle(r)

bps = 200
bs = 32
print('ideal:', 1/(bps*bs)*1000, 's for reading 1000 images')

if os.path.exists('tmp'):shutil.rmtree('tmp')
os.mkdir('tmp')


dat = []
for f in files:
    filename = f.split('/')[-1].split('.')[0]
    img = cv2.imread(f)
    dat.append(img)
    cv2.imwrite('tmp/'+filename+'.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    cv2.imwrite('tmp/'+filename+'.bmp', img)

files = sorted(list(glob.glob('tmp/*.jpg')))
t = time.time()
for f in range(len(files)):
    img = cv2.imread(files[r[f]])
print('jpg:',(time.time()-t)/len(files)*1000,'s')

files = sorted(list(glob.glob('tmp/*.bmp')))
t = time.time()
for f in range(len(files)):
    img = cv2.imread(files[r[f]])
print('bmp:',(time.time()-t)/len(files)*1000,'s')

if os.path.isfile('tmp/dat.h5'):os.remove('tmp/dat.h5')
with h5py.File('tmp/tmp.h5') as f:
    f.create_dataset('img', data = np.array(dat))
    f.create_dataset('img_chunk', data = np.array(dat), chunks=(1, *dat[0].shape))
    f.create_dataset('img_chunk_small', data = np.array(dat), chunks=(1, 64,64,3))
    f.create_dataset('img_chunk_comp', data = np.array(dat), chunks=(1, *dat[0].shape), compression="gzip", compression_opts=9)
    f.create_dataset('img_chunk_comp_small', data = np.array(dat), chunks=(1, 64, 64, 3), compression="gzip", compression_opts=9)

rnd_str = 'tmp/'+(''.join(random.choice(string.ascii_lowercase) for i in range(12)))+'.h5'
os.rename('tmp/tmp.h5', rnd_str)

dat = h5py.File(rnd_str)['img']
t = time.time()
for idx in range(dat.shape[0]):
    img = dat[r[idx]]
print('h5: ',(time.time()-t)/len(files)*1000,'s')

h5file = h5py.File(rnd_str)
dat = h5file['img_chunk']
t = time.time()
for idx in range(dat.shape[0]):
    img = dat[r[idx]]
print('h5c:',(time.time()-t)/len(files)*1000,'s')

dat = h5file['img_chunk_small']
t = time.time()
for idx in range(dat.shape[0]):
    img = dat[r[idx]]
print('h5cs:',(time.time()-t)/len(files)*1000,'s')

dat = h5file['img_chunk_comp']
t = time.time()
for idx in range(dat.shape[0]):
    img = dat[r[idx]]
print('h5cc:',(time.time()-t)/len(files)*1000,'s')

dat = h5file['img_chunk_comp_small']
t = time.time()
for idx in range(dat.shape[0]):
    img = dat[r[idx]]
print('h5ccs:',(time.time()-t)/len(files)*1000,'s')

h5file.close()

shutil.rmtree('tmp')
