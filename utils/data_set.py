from torch.utils.data import Dataset
import pandas as pd

class CloudDataSet(Dataset):
  def __init__(self, csv_file, root_dir,transform=None):
        self.set_data = pd.read_csv(csv_file)
        self.root_dir =  root_dir
        self.transform = transform

  def __len__(self):
    return len(self.set_data)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()
    
    img_id = self.set_data.iloc[idx,0]
    image = Image.open(self.root_dir + self.set_data.iloc[idx, 0])
    mask = []
    idex = 1
    for i in range (4):
      mk = rle_to_mask(self.set_data.iloc[idx, idex])
      mk = PIL.Image.fromarray(mk)
      mask.append(mk)
      idex += 1
    
    if self.transform:
      image = self.transform(image)
      mask = [self.transform(mask[i]).squeeze() for i in range(4)]
      mask = torch.stack(mask)
    
    sample = {'image':image,'mask':mask}
    return sample