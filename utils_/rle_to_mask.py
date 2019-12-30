def rle_to_mask(rle_string, height=1400, width=2100):
  rows, cols = height, width
  if rle_string == '-1':
        return np.zeros((height, width),dtype=np.uint8)
  else:
        rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]
        rle_pairs = np.array(rle_numbers).reshape(-1,2)
        img = np.zeros(rows*cols, dtype=np.uint8)
        for index, length in rle_pairs:
            index -= 1
            img[index:index+length] = 1
        img = img.reshape(rows,cols, order='F')
        return img
