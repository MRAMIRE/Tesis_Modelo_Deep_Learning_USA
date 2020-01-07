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
  a = np.where(mask == 'Fish')
  a=mask[a[0],0]
  m_fish = rle_to_mask(a[0])
  a = np.where(mask == 'Flower')
  a=mask[a[0],0]
  m_flower = rle_to_mask(a[0])
  a = np.where(mask == 'Gravel')
  a=mask[a[0],0]
  m_gravel = rle_to_mask(a[0])
  a = np.where(mask == 'Sugar')
  a=mask[a[0],0]
  m_sugar = rle_to_mask(a[0])
  masks = np.array([m_fish, m_flower,m_gravel,m_sugar])
  return masks
