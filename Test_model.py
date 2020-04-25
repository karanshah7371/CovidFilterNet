from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from os import sys
import os
import os.path
import glob

img_width, img_height = 200, 250

model = load_model('./Covidx.h5')
img_dir = "./Test/xyar_bact.jpg"
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
#for f1 in files:
img = image.load_img(img_dir, target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = model.predict_classes(images, batch_size=1)

print("     \n      ")
    
if classes[0] == 0:
    print('covid')
    print(model.predict_proba(images, batch_size=1)[:,0])
        
elif classes[0] == 1:
    print('Non Covid P')
    print(model.predict_proba(images, batch_size=1)[:,1])

else:
    print('Normal')
    print(model.predict_proba(images, batch_size=1)[:,2])
        
