#!/usr/bin/env python
# coding: utf-8

# In[14]:


import keras
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers import Input
from keras.models import Model
from keras.regularizers import *
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from os import listdir
from os.path import isfile, join
from keras import *
from keras.models import load_model
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# In[37]:


img_width = 200
img_height = 250

train_data_dir = './Train'
validation_data_dir = './Validate'
train_samples = 3420
validation_samples = 482
epochs = 15
batch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = (3,img_width, img_height)
    print("C.F")
else:
    input_shape = (img_width, img_height,3)
    print("C.L")

# In[38]:


model = Sequential()

model.add(Conv2D(128, (2, 2),input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(128, (2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(128, (2, 2)))
model.add(Activation('linear'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(64, (2, 2)))
model.add(Activation('linear'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(32, (2, 2)))
model.add(Activation('linear'))
model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(32, (2, 2)))
model.add(Activation('linear'))
model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) 
model.add(Dense(2048))
model.add(Activation('linear'))

model.add(Activation('relu'))
model.add(Dense(1024))
model.add(Activation('linear'))

model.add(BatchNormalization())
model.add(Activation('relu'))
          
model.add(Dropout(0.3))
model.add(Dense(1024))
model.add(Activation('linear'))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Activation('softmax'))


# In[39]:


import keras
from keras import optimizers
model.compile(loss='categorical_crossentropy', 
              optimizer=keras.optimizers.Adam(lr=.0001,beta_2=0.999,beta_1=0.90),
              metrics=['accuracy'])


# In[40]:


train_datagen = ImageDataGenerator()


# In[41]:


test_datagen = ImageDataGenerator()


# In[42]:


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


# In[43]:


print(train_generator.class_indices)


# In[44]:


imgs, labels = next(train_generator)


# In[45]:


validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


# In[ ]:



history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_samples // batch_size,
    shuffle=True
    )
model.save('Covidx.h5') 

print('Successful :-) --737--')
