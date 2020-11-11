#!/usr/bin/env python
# coding: utf-8

# # EAS 596 Fundamental of Artificial Intelligence

# # Dhruv Patel(#50321707)

# ## Homework 3 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from numpy import array

import keras
from keras.models import Sequential, Model
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


location = 'https://drive.google.com/a/buffalo.edu/uc?export=download&confirm=aTCV&id=1w6zxPR-uJiR0LVPwACvE17NHnx_eIFgx'
path_to_zip = tf.keras.utils.get_file('USPSdata.zip', origin=location, extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'USPSdata')


# In[3]:


train_USPS = os.path.join(PATH, 'Train')
test_USPS = os.path.join(PATH, 'Test')


# In[4]:


img_rows = 28
img_cols = 28

batch_size = 128
num_classes = 10
epochs = 50


# #### Train Data Processing 

# In[5]:


CATEGORY_TRAIN = ['0','1','2','3','4','5','6','7','8','9']


# In[6]:


training_data = []

def create_training_data():
    for category in CATEGORY_TRAIN:
        path = os.path.join(train_USPS,category)
        class_num = CATEGORY_TRAIN.index(category)
        for imag in os.listdir(path):
            try:
                imag_array = cv2.imread(os.path.join(path,imag),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(imag_array,(img_rows,img_cols))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
create_training_data()


# In[7]:


for category in CATEGORY_TRAIN:
    path = os.path.join(train_USPS,category)
    class_num = CATEGORY_TRAIN.index(category)
    for imag in os.listdir(path):
            
        imag_array = cv2.imread(os.path.join(path,imag),cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(imag_array,(img_rows,img_cols))
        training_data.append([new_array, class_num])
        plt.imshow(imag_array,cmap = 'gray')
        plt.title(class_num)
        plt.colorbar()
        plt.show()
        
        break


# In[8]:


print(len(training_data))


# In[9]:


print(len(training_data))


# In[10]:


import random
random.shuffle(training_data)


# In[11]:


X_train_USPS = []
y_train_USPS = []


# In[12]:


for features, label in training_data:
    X_train_USPS.append(features)
    y_train_USPS.append(label)


# #### Test Data Processing 

# In[16]:


test_img = os.listdir(test_USPS) # converting test imagies into list

test_0_img = test_img[1350:1500]
test_1_img = test_img[1200:1350]
test_2_img = test_img[1050:1200]
test_3_img = test_img[900:1050]
test_4_img = test_img[750:900]
test_5_img = test_img[600:750]
test_6_img = test_img[450:600]
test_7_img = test_img[300:450]
test_8_img = test_img[150:300]
test_9_img = test_img[0:150]
print('type of file:',type(test_USPS) )
print('type of file:',type(test_0_img) )

num_0_test = len(test_0_img)
num_1_test = len(test_1_img)
num_2_test = len(test_2_img)
num_3_test = len(test_3_img)
num_4_test = len(test_4_img)
num_5_test = len(test_5_img)
num_6_test = len(test_6_img)
num_7_test = len(test_7_img)
num_8_test = len(test_8_img)
num_9_test = len(test_9_img)

print('Number of images of 0:',num_0_test)
print('Number of images of 1:',num_1_test)
print('Number of images of 2:',num_2_test)
print('Number of images of 3:',num_3_test)
print('Number of images of 4:',num_4_test)
print('Number of images of 5:',num_5_test)
print('Number of images of 6:',num_6_test)
print('Number of images of 7:',num_7_test)
print('Number of images of 8:',num_8_test)
print('Number of images of 9:',num_9_test)


# In[17]:


CATEGORIES_TEST = [test_0_img,test_1_img,test_2_img,test_3_img,test_4_img,test_5_img,test_6_img,test_7_img,test_8_img,test_9_img]
testing_data=[]
def create_test_data():
     for categories in CATEGORIES_TEST:
        class_num_test = CATEGORIES_TEST.index(categories)
        for img in categories:
            try:
                img_array_test = cv2.imread(os.path.join(test_USPS,img),cv2.IMREAD_GRAYSCALE)
                new_array_test = cv2.resize(img_array_test,(img_rows,img_cols))
                testing_data.append([new_array_test, class_num_test])
            except Exception as e:
                pass
        
        
create_test_data()


# In[18]:


for categories in CATEGORIES_TEST:
    class_num_test = CATEGORIES_TEST.index(categories)
    for img in categories:
        img_array_test = cv2.imread(os.path.join(test_USPS,img),cv2.IMREAD_GRAYSCALE)
        new_array_test = cv2.resize(img_array_test,(img_rows,img_cols))
        plt.imshow(img_array_test,cmap = 'gray')
        plt.title(class_num_test)
        plt.colorbar()
        plt.show()
        break 
         


# In[19]:


print(len(testing_data))


# In[20]:


import random
random.shuffle(testing_data)


# In[21]:


X_test_USPS = []
y_test_USPS = []


# In[22]:


for features, label in training_data:
    X_test_USPS.append(features)
    y_test_USPS.append(label)


# In[23]:


# Converting List into Array
X_train_USPS = array(X_train_USPS)
X_test_USPS = array(X_test_USPS)
y_train_USPS = array(y_train_USPS)
y_test_USPS = array(y_test_USPS)
print('USPS X_train shape:',X_train_USPS.shape)
print('USPS X_test shape:',X_test_USPS.shape)
print('USPS y_train shape:',y_train_USPS.shape)
print('USPS y_test shape:',y_test_USPS.shape)


# In[24]:


print('Lenth:')
print('Lenth of USPS X_train:',len(X_train_USPS))
print('Lenth of USPS X_test:',len(X_test_USPS))
print('Lenth of USPS y_train:',len(y_train_USPS))
print('Lenth of USPS y_test',len(y_test_USPS))
print('\n')
print('Type:')
print(type(X_train_USPS))
print(type(X_test_USPS))
print(type(y_train_USPS))
print(type(y_test_USPS))


# In[25]:


if K.image_data_format() == 'channels_first':
    X_train_USPS = X_train_USPS.reshape(X_train_USPS.shape[0], 1, img_rows, img_cols)
    X_test_USPS = X_test_USPS.reshape(X_test_USPS.shape[0], 1, img_rows, img_cols)
    input_shape_USPS = (1, img_rows, img_cols)
else:
    X_train_USPS = X_train_USPS.reshape(X_train_USPS.shape[0], img_rows, img_cols, 1)
    X_test_USPS = X_test_USPS.reshape(X_test_USPS.shape[0], img_rows, img_cols, 1)
    input_shape_USPS = (img_rows, img_cols, 1)

X_train_USPS = X_train_USPS.astype('float32')
X_test_USPS = X_test_USPS.astype('float32')
X_train_USPS /= 255
X_test_USPS /= 255
print('x_train shape:', X_train_USPS.shape)
print('y_train shape:', y_train_USPS.shape)
print('x_test shape:', X_test_USPS.shape)
print('y_test shape:', y_test_USPS.shape)
print(X_train_USPS.shape[0], 'train samples')
print(X_test_USPS.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train_USPS = keras.utils.to_categorical(y_train_USPS, num_classes)
y_test_USPS = keras.utils.to_categorical(y_test_USPS, num_classes)


# # MNIST Data

# In[26]:


# the data, split between train and test sets
(X_train_MNIST, y_train_MNIST), (X_test_MNIST, y_test_MNIST) = mnist.load_data()

print('MNIST x_train Type:', type(X_train_MNIST))
print('MNIST y_train Type:', type(y_train_MNIST))
print('MNIST x_test Type:', type(X_test_MNIST))
print('MNIST y_test Type:', type(y_test_MNIST))
print('\n')
print('MNIST x_train Shape:', X_train_MNIST.shape)
print('MNIST y_train Shape:', y_train_MNIST.shape)
print('MNIST x_test  Shape:', X_test_MNIST.shape)
print('MNIST y_test  Shape:', y_test_MNIST.shape)


# In[27]:


if K.image_data_format() == 'channels_first':
    X_train_MNIST = X_train_MNIST.reshape(X_train_MNIST.shape[0], 1, img_rows, img_cols)
    X_test_MNIST = X_test_MNIST.reshape(X_test_MNIST.shape[0], 1, img_rows, img_cols)
    input_shape_MNIST = (1, img_rows, img_cols)
else:
    X_train_MNIST = X_train_MNIST.reshape(X_train_MNIST.shape[0], img_rows, img_cols, 1)
    X_test_MNIST = X_test_MNIST.reshape(X_test_MNIST.shape[0], img_rows, img_cols, 1)
    input_shape_MNIST = (img_rows, img_cols, 1)

X_train_MNIST = X_train_MNIST.astype('float32')
X_test_MNIST = X_test_MNIST.astype('float32')
X_train_MNIST /= 255
X_test_MNIST /= 255
print('x_train shape:', X_train_MNIST.shape)
print('y_train shape:', y_train_MNIST.shape)
print('x_test shape:', X_test_MNIST.shape)
print('y_test shape:', y_test_MNIST.shape)
print(X_train_MNIST.shape[0], 'train samples')
print(X_test_MNIST.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train_MNIST = keras.utils.to_categorical(y_train_MNIST, num_classes)
y_test_MNIST = keras.utils.to_categorical(y_test_MNIST, num_classes)


# # Step 3

# ### Model A 

# #### Create Model 

# In[28]:



model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape_USPS))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


# #### Compile the Model

# In[29]:


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])


# In[30]:


model.summary()


# In[31]:


model.fit(X_train_USPS, y_train_USPS,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test_USPS, y_test_USPS))
score = model.evaluate(X_test_USPS, y_test_USPS, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Validation accuracy: 94.71%,
# Test loss: 17.06%,
# Test accuracy: 94.70%

# In[32]:


y_pred_USPS = model.predict(X_train_USPS, batch_size=None, verbose=0)


# ### Confusion Matrix

# In[33]:


confusion_matric = confusion_matrix(y_test_USPS.argmax(axis=1), y_pred_USPS.argmax(axis=1))
confusion_matric


# ### Model B

# In[34]:


train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')


# #### Create Model

# In[35]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape_MNIST))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


# #### Compile the model

# In[36]:


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])
model.summary()


# In[37]:


model.fit(X_train_MNIST, y_train_MNIST,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test_USPS, y_test_USPS))
score = model.evaluate(X_test_USPS, y_test_USPS, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# validation accuracy: 9.99%,
# Test loss: 2766.68%
# Test accuracy: 9.99%

# In[38]:


y_pred_USPS = model.predict(X_test_USPS, batch_size=None, verbose=0)
y_pred_USPS.shape


# In[39]:


confusion_matric = confusion_matrix(y_test_USPS.argmax(axis=1), y_pred_USPS.argmax(axis=1))
confusion_matric


# ### ==> Here test accuracy is too low because number of test images are too low compared to the trained images. Therefore, model is overfititng the test images which turn out to be low test accuracy. 

# ### =>Some Issues with Neural Network: 
# ##### 1.Sometimes neural networks fail to converge due to low dimensionality. 
# ##### 2.Even a small change in weights can lead to significant change in output. 
# ##### 3.sometimes results may be worse. The gradient may become zero .
# ##### 4.In this case , weight optimization fails. Data overfitting. 
# ##### 5.Time complexity is too high. Sometimes algorithm runs for days even on small data set. 
# ##### 6.We get the same output for every input when we predict.

# ### Model C

# #### Concatenating MNIST and USPS datasets

# In[40]:


X_train_MNIST_USPS = np.concatenate((X_train_MNIST, X_train_USPS), axis=0)
y_train_MNIST_USPS = np.concatenate((y_train_MNIST, y_train_USPS), axis=0)


# In[41]:


print('MNIST x_train Type:', type(X_train_MNIST_USPS))
print('MNIST y_train Type:', type(y_train_MNIST_USPS))
print('MNIST x_test Type:', type(X_test_USPS))
print('MNIST y_test Type:', type(y_test_USPS))
print('\n')
print('Total MNIST and USPS x_train Shape:', X_train_MNIST_USPS.shape)
print('Total MNIST and USPS y_train Shape:', y_train_MNIST_USPS.shape)
print('Total MNIST and USPS x_test  Shape:', X_test_USPS.shape)
print('Total MNIST and USPS y_test  Shape:', y_test_USPS.shape)


# In[42]:


if K.image_data_format() == 'channels_first':
    X_train_MNIST_USPS = X_train_MNIST_USPS.reshape(X_train_MNIST_USPS.shape[0], 1, img_rows, img_cols)
    X_test_USPS = X_test_USPS.reshape(X_test_USPS.shape[0], 1, img_rows, img_cols)
    input_shape_MNIST_USPS = (1, img_rows, img_cols)
else:
    X_train_MNIST_USPS = X_train_MNIST_USPS.reshape(X_train_MNIST_USPS.shape[0], img_rows, img_cols, 1)
    X_test_USPS = X_test_USPS.reshape(X_test_USPS.shape[0], img_rows, img_cols, 1)
    input_shape_MNIST_USPS = (img_rows, img_cols, 1)

X_train_MNIST_USPS = X_train_MNIST_USPS.astype('float32')
X_test_USPS = X_test_USPS.astype('float32')

print('x_train of MNIST and USPS shape:', X_train_MNIST_USPS.shape)
print('y_train of MNIST and USPS shape:', y_train_MNIST_USPS.shape)
print('x_test shape of USPS:', X_test_USPS.shape)
print('y_test shapeof USPS:', y_test_USPS.shape)
print(X_train_MNIST_USPS.shape[0], 'train samples of MNIST and USPS')
print(X_test_USPS.shape[0], 'test samples of USPS')


# #### Create Model

# In[43]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape_MNIST_USPS))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


# #### Compile Model

# In[44]:


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])
model.summary()


# In[45]:


model.fit(X_train_MNIST_USPS, y_train_MNIST_USPS,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test_USPS, y_test_USPS))
score = model.evaluate(X_test_USPS, y_test_USPS, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Validation accuracy: 92.11%,
# Test loss: 23.42%,
# Test accuracy: 92.10%

# In[46]:


y_pred_USPS = model.predict(X_test_USPS, batch_size=None, verbose=0)


# In[47]:


confusion_matric = confusion_matrix(y_test_USPS.argmax(axis=1), y_pred_USPS.argmax(axis=1))
confusion_matric


# #  Tuning hyper-parameters:

# ## =>With 3 hidden layer 

# #### Step 3 repeatation

# In[48]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape_USPS))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


# In[49]:


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])
model.summary()


# In[51]:


model.fit(X_train_USPS, y_train_USPS,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test_USPS, y_test_USPS))
score = model.evaluate(X_test_USPS, y_test_USPS, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Training Accuracy : 98%,
# Testing Accuracy  : 87%,
# Test Loss         : 9%

# In[52]:


y_pred_USPS = model.predict(X_train_USPS, batch_size=None, verbose=0)
y_pred_USPS.shape


#  

# In[53]:


confusion_matric = confusion_matrix(y_test_USPS.argmax(axis=1), y_pred_USPS.argmax(axis=1))
confusion_matric


# #### Step 4 Repeatation

# In[88]:


model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape_USPS))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])
model.summary()


# In[89]:


X_train_MNIST[0:20000].shape


# In[90]:


X_test_USPS.shape


# In[91]:


model.fit(X_train_MNIST[0:20009], y_train_MNIST[0:20009],
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test_USPS, y_test_USPS))
score = model.evaluate(X_test_USPS, y_test_USPS, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# val_accuracy: 8.07%,
# Test loss: 2836.59%,
# Test accuracy: 8.06%

# In[92]:


y_pred_USPS = model.predict(X_test_USPS, batch_size=None, verbose=0)


# In[93]:


confusion_matric = confusion_matrix(y_test_USPS.argmax(axis=1), y_pred_USPS.argmax(axis=1))
confusion_matric


# #### Step 5 Repeatation

# In[94]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape_USPS))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu')) #Hidden Layer 1
model.add(Dense(128, activation='relu')) #Hidden Layer 2
model.add(Dense(128, activation='relu')) #Hidden Layer 3
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])
model.summary()


# In[96]:


model.fit(X_train_MNIST_USPS, y_train_MNIST_USPS,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test_USPS, y_test_USPS))
score = model.evaluate(X_test_USPS, y_test_USPS, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# val_accuracy: 98.57%,
# Test loss: 5.36%,
# Test accuracy: 98.56%

# In[97]:


y_pred_USPS = model.predict(X_test_USPS, batch_size=None, verbose=0)


# In[98]:


confusion_matric = confusion_matrix(y_test_USPS.argmax(axis=1), y_pred_USPS.argmax(axis=1))
confusion_matric


# # =>With 5 hidden layer

# ### Step 3

# In[65]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape_USPS))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu')) # Hidden Layer 1
model.add(Dense(128, activation='relu')) # Hidden Layer 2
model.add(Dense(128, activation='relu')) # Hidden Layer 3
model.add(Dense(128, activation='relu')) # Hidden Layer 4
model.add(Dense(128, activation='relu')) # Hidden Layer 5

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])
model.summary()


# In[66]:


model.fit(X_train_USPS, y_train_USPS,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test_USPS, y_test_USPS))
score = model.evaluate(X_test_USPS, y_test_USPS, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# validation accuracy: 98.53%,
# Test loss: 5.66%,
# Test accuracy: 98.53%

# In[67]:


y_pred_USPS = model.predict(X_train_USPS, batch_size=None, verbose=0)


# In[ ]:


confusion_matric = confusion_matrix(y_test_USPS.argmax(axis=1), y_pred_USPS.argmax(axis=1))
confusion_matric


# ### Step 4

# In[68]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape_USPS))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu')) # Hidden Layer 1
model.add(Dense(128, activation='relu')) # Hidden Layer 2
model.add(Dense(128, activation='relu')) # Hidden Layer 3
model.add(Dense(128, activation='relu')) # Hidden Layer 4
model.add(Dense(128, activation='relu')) # Hidden Layer 5

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])
model.summary()


# In[70]:


model.fit(X_train_MNIST, y_train_MNIST,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test_USPS, y_test_USPS))
score = model.evaluate(X_test_USPS, y_test_USPS, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# val_accuracy: 7.10%,
# Test loss: 2550.80%,
# Test accuracy: 7.09%

# In[71]:


y_pred_USPS = model.predict(X_test_USPS, batch_size=None, verbose=0)


# In[72]:


confusion_matric = confusion_matrix(y_test_USPS.argmax(axis=1), y_pred_USPS.argmax(axis=1))
confusion_matric


# ### Step 5

# In[73]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape_USPS))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu')) # Hidden Layer 1
model.add(Dense(128, activation='relu')) # Hidden Layer 2
model.add(Dense(128, activation='relu')) # Hidden Layer 3
model.add(Dense(128, activation='relu')) # Hidden Layer 4
model.add(Dense(128, activation='relu')) # Hidden Layer 5

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])
model.summary()


# In[74]:


model.fit(X_train_MNIST_USPS, y_train_MNIST_USPS,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test_USPS, y_test_USPS))
score = model.evaluate(X_test_USPS, y_test_USPS, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Validation accuracy: 99.67%
# Test loss: 1.88%
# Test accuracy: 99.67%

# In[75]:


y_pred_USPS = model.predict(X_test_USPS, batch_size=None, verbose=0)


# In[76]:


confusion_matric = confusion_matrix(y_test_USPS.argmax(axis=1), y_pred_USPS.argmax(axis=1))
confusion_matric


# # Observations:

# ==>From the above experiment, It is be concluded that increased hidden layer result in better accuracy(5 Hidden layer has better accuracy than 3 hidden layer).

# ==> Also, adding optimal nodes(neurons) can also help in buildin better model

# ==> The issue with Model B is overfitting of data. That is due to inbalance between training parameters and testing parameters. This problem can also be solved by adding 'Augmented Images'. For example, adding similar images with different view of angle and view(Mirrored, Tilted)

# In[ ]:




