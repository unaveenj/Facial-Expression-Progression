import cv2
import numpy as np
train_data = list(np.load('Data.npy' , allow_pickle = True))
print(len(train_data))
for image,out in train_data:
    image = cv2.resize(image , (300,300))
    print(image.shape)
    cv2.imshow('img',image)
    print(out)
    
    #print(choice)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()

x = [i[0] for i in train_data]
y = [i[0] for i in train_data]
print(len(x),len(y))
