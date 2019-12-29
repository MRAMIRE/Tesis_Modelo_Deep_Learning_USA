class CloudDataSet(Dataset):
  

  def __init__(self, csv_file, root_dir, transform=None):
        self.set_data = pd.read_csv(csv_file)
        self.root_dir =  root_dir
        self.transform = transform

  def __len__(self):
    return len(self.set_data)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()
    
    img_id = self.set_data.iloc[idx,0]
    image = io.imread(self.root_dir + self.set_data.iloc[idx, 0])
    #
    image = cv2.resize(image, (int(2100/2),int(1400/2)))
    #
    image = torch.from_numpy(image)
    image = image.permute(2,0,1)

    mask = []
    idex = 1
    for i in range(4):
      mk= rle_to_mask(self.set_data.iloc[idx, idex])
      #
      mk = cv2.resize(mk, (int(2100/2),int(1400/2)))
      #
      mk = torch.from_numpy(mk)
      mask.append(mk)
      idex += 1
    mask = torch.stack(mask)
    
    sample = {'image':image,'mask':mask}
    return sample
