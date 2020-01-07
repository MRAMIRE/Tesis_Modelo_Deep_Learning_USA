import numpy as np
from skimage import io, transform
import pylab as plt
from .rle_to_mask import *

def plot_img_masks(img,mask,root,alpha = 0.5):
 pri = []
 for m in range(4):
   color = (0,0,1)
   img = io.imread(root + image)
   for c in range(3):
     img[:, :, c] = np.where(mask[m] == 1,img[:, :, c] *(1 - alpha) + alpha * color[c] * 255,img[:, :, c])
   pri.append(img)

 f, ax = plt.subplots(1, 4, figsize=(21,14))
 ax[0].imshow(pri[0])
 ax[0].set_title('Fish Mask')
 ax[0].axis('off')
 ax[1].imshow(pri[1])
 ax[1].set_title('Flower Mask')
 ax[1].axis('off')
 ax[2].imshow(pri[2])
 ax[2].set_title('Gravel Mask')
 ax[2].axis('off')
 ax[3].imshow(pri[3])
 ax[3].set_title('Sugar Mask')
 ax[3].axis('off')