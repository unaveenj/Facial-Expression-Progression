import cv2,os,sys
import numpy as np
from tqdm import tqdm
from random import shuffle
data_dir = 'Dataset/'
directories = [i for i in os.listdir(data_dir)]
data = []
for i in tqdm(os.listdir(data_dir)):
    path = data_dir+i
    class_ = directories.index(i)
    for j in os.listdir(path):
        img = cv2.imread(os.path.join(path,j))
        img = cv2.resize(img , (224,224))
        data.append([img , class_])
shuffle(data)
np.save('Data.npy',data)

x = [data_[0] for data_ in data]
y = [data_[1] for data_ in data]
print(len(y))