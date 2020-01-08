import torch
import numpy as np
import copy
import pylab as plt
from torchvision import transforms

def real_vs_pred(muestra,val_dataset,net,alpha,colorPred = (1,0,0),colorReal = (0.5,1,0)):
  
  #Imagen
  image = val_dataset[muestra]['image']
  image = transforms.ToPILImage()(image).convert("RGB")
  image = np.array(image)

  #Mascaras Predicción
  img_to_net = val_dataset[muestra]['image'].unsqueeze(0)
  pred = net(img_to_net)
  pred = pred.squeeze(0)
  pred = torch.sigmoid(pred)
  pred = torch.where(pred>=0.5,torch.Tensor([1]),torch.Tensor([0]))
  maskPred = [pred[i,:,:].squeeze().detach().numpy() for i in range(4)]

  #Mascaras Real
  mask_Real = val_dataset[muestra]['mask']
  maskReal = [mask_Real[i,:,:].squeeze().detach().numpy() for i in range(4)]

  pri = []
  for m in range(4):
    imgs = copy.copy(image)
    for c in range(3):
      imgs[:, :, c] = np.where(maskPred[m] == 1,imgs[:, :, c] *(1 - alpha) + alpha * colorPred[c] * 255,imgs[:, :, c])
    for c in range(3):
      imgs[:, :, c] = np.where(maskReal[m] == 1,imgs[:, :, c] *(1 - alpha) + alpha * colorReal[c] * 255,imgs[:, :, c])
    pri.append(imgs)

  f, ax = plt.subplots(1, 4, figsize=(21,14))
  title = ['Fish Mask','Flower Mask','Gravel Mask','Sugar Mask']

  for i in range(4):
    ax[i].imshow(pri[i])
    ax[i].set_title(title[i])
    ax[i].axis('off')