import numpy as np

def get_mask(image_id,data_frame):
  mk = []
  index = 0
  for i in data_frame['Id']:
    if i == image_id:
      t = data_frame['EncodedPixels'][index]
      mk.append(t)
      t = data_frame['Label'][index]
      mk.append(t)
    mask = np.array(mk).reshape(-1,2)
    index += 1
  return mask
