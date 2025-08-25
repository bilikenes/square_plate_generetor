import pylab as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

img = np.uint8(mpimg.imread(r"D:\Work\python_files\LicensePlateGenerator\plates\realistic_plates\01AAC777.png"))

img = np.uint8((0.2126* img[:,:,0]) + \
np.uint8(0.7152 * img[:,:,1]) +\
np.uint8(0.0722 * img[:,:,2]))

threshold = 64

it = np.nditer(img, flags=['multi_index'], op_flags=['writeonly'])
while not it.finished:
    if it[0] > threshold:
        it[0] = threshold
    else:
        it[0] = 0
    it.iternext()

im = Image.fromarray(img)
im.save("output.jpeg")
plt.imshow(img,cmap=plt.cm.gray)
plt.show()