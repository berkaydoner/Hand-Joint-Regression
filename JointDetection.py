

from google.colab import drive
drive.mount('/content/drive')

!unzip /content/drive/My\ Drive/RovitDatasetBig_.zip\ adlı\ dosyanın\ kopyası -d /content/Data

# Commented out IPython magic to ensure Python compatibility.
# %cd Data

# Commented out IPython magic to ensure Python compatibility.
# %cd Rovit\ Dataset

import pickle
import os
from os import listdir
from os.path import isfile,join

bounding_boxes = {}
images = {}
projections= {}
for i in range(21):
  mypath = "/data_"+str(i+1)
  if(i!=17): #there is an problem with dataset 18, discard it
    bounding_boxes[mypath]=["bounding_boxes"+mypath+"/"+ f for f in listdir("bounding_boxes"+mypath) if isfile(join("bounding_boxes"+mypath, f))]
    images[mypath]=["annotated_frames"+mypath+"/"+f for f in listdir("annotated_frames"+mypath) if isfile(join("annotated_frames"+mypath, f)) and join("annotated_frames"+mypath, f).endswith('.jpg')]
    projections[mypath]=["projections_2d"+ mypath +"/" + f for f in listdir("projections_2d"+mypath) if isfile(join("projections_2d"+mypath, f))]

import keras
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

length = 0
width = 0
for i in range(21):
  if(i!=17): # there is a problem with dataset 18, discard it
    num_examples = len(bounding_boxes["/data_"+str(i+1)])
    bounding_boxes["/data_"+str(i+1)].sort()
    bounding_array = np.zeros(shape=(num_examples,4)) #there are 4 values associated with an example
    for j in range(num_examples):
      a=open(bounding_boxes["/data_"+str(i+1)][j], "r").read().split()
      length = max(length,int(a[5])-int(a[1]))
      width = max(width,int(a[7])-int(a[3]))# take the bounding box length and width
      del a[0:7:2]
      bounding_array[j]=a 
    pickle.dump(bounding_array, open( "bounding_array"+str(i+1)+".p", "wb" ) )# save the pickle for later use

#maximum length and width of the bounding boxes are going to be used to pad the cropped images
print(length)
print(width)
print(bounding_array[0])

tmp_image = load_img(images["/data_12"][6])
tmp_img_array = img_to_array(tmp_image)
image_sizes = np.shape(tmp_img_array)
print(image_sizes[0],image_sizes[1],image_sizes[2])
#an example to find the size of the images

import tensorflow.image
import tensorflow as tf

for i in range(21):
  if i==0 or i==4 or i==5 or i==13:
    # Since arrays use a lot of memory, I decided to choose 4 datasets only, they contain almost 8000 examples in total.
    num_examples=len(images["/data_"+str(i+1)])
    images["/data_"+str(i+1)].sort()
    img_array=np.zeros(shape=(num_examples,332,382,3))# All images will be in shape 332x382x3 which is the maximum bounding box shape
    print(np.shape(img_array))
    pickle_in = open("bounding_array"+str(i+1)+".p","rb")
    bound = pickle.load(pickle_in)
    for j in range(num_examples):
      tmp = load_img(images["/data_"+str(i+1)][j])
      a = bound[j]
      x=abs(int(a[0]))
      y=abs(int(a[1]))
      h=abs(int(a[2])-int(a[0]))
      w=abs(int(a[3])-int(a[1]))
      arr = img_to_array(tmp)
      img_array[j,0:h,0:w,0:3]=arr[x:x+h,y:y+w,0:3] # This allows us to pad images to the shape 332x382
    for k in range(int(num_examples/256)):#The datasets are separated into batches for training, since pickle allows only 2gb data, I couldn't pickle the whole set 
      pickle.dump(img_array[k*256:(k+1)*256], open( "img_array"+str(i+1)+"*"+str(k)+".p", "wb" ) )

def trainset_generator(dataset_num):
  num_examples=int(len(images["/data_"+str(dataset_num)])/256)*256
  trainset = num_examples*[]
  for i in range(int(num_examples/256)):
    pickle_in = open("img_array"+str(dataset_num)+"*"+str(i)+".p","rb")
    imgarr = pickle.load(pickle_in)
    for j in range(256):
      trainset.append(np.array(imgarr[j]))
  trainset = np.array(trainset)
  trainset/=255
  return trainset

import matplotlib.pyplot as plt
trainset = trainset_generator(1)
plt.imshow(trainset[0])

def func(p):#This method is for shifting the coordinates of the joints, this is necessary since we used padding
  num_examples=len(bounding_boxes["/data_"+str(p)])
  projections["/data_"+str(p)].sort()
  proj_array = np.zeros(shape=(num_examples,42))
  pickle_in = open("bounding_array"+str(p)+".p","rb")
  bound = pickle.load(pickle_in)  
  projections["/data_"+str(p)].sort()
  for i in range(num_examples):
    a=open(projections["/data_"+str(p)][i], "r").read().split()
    del a[0:61:3]
    proj_array[i]=np.asarray([float(f) for f in a])
    for j in range(len(proj_array[i])):
      if j%2==0:
        proj_array[i][j] -= bound[i][1]  
      else:
        proj_array[i][j] -= bound[i][0]
  for k in range(int(num_examples/256)):
    pickle.dump(proj_array[k*256:(k+1)*256], open( "proj_array"+str(p)+"*"+str(k)+".p", "wb" ) )

func(1)

def labelset_generator(dataset_num):
  num_examples=int(len(projections["/data_"+str(dataset_num)])/256)*256
  labelset = num_examples*[]
  for i in range(int(num_examples/256)):
    pickle_in = open("proj_array"+str(dataset_num)+"*"+str(i)+".p","rb")
    proj_array = pickle.load(pickle_in)
    for j in range(256):
      labelset.append(np.array(proj_array[j]))
  labelset = np.array(labelset)
  return labelset

labelset = labelset_generator(1)
print(labelset.shape)
print(trainset.shape)

implot = plt.imshow(trainset[2])
for x in range(42):
  if(x%2==0):
    plt.scatter([labelset[2][x]],[labelset[2][x+1]],color="red")

from keras.layers import Conv2D, BatchNormalization, Dropout, Input, Dense, Flatten, MaxPool2D, GlobalAveragePooling2D
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.models import Model

###Transfer Learning Model
base_model=VGG16(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.
x=base_model.output
x = GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(42,activation='linear')(x) #final layer with softmax activation
model = Model(inputs=base_model.input, outputs=preds)
for layer in model.layers[:18]:
    layer.trainable=False

###Basic Model
cifNet = Sequential()

#Layer1
cifNet.add(BatchNormalization())
cifNet.add(Conv2D(32,(5,5),activation='relu',padding='valid'))
cifNet.add(MaxPool2D())
cifNet.add(Dropout(0.2))

#Layer2
cifNet.add(BatchNormalization())
cifNet.add(Conv2D(64,(5,5),activation='relu',padding='valid'))
cifNet.add(MaxPool2D())
cifNet.add(Dropout(0.2))

#Layer3
cifNet.add(BatchNormalization())
cifNet.add(Conv2D(128,(5,5),activation='relu',padding='valid'))
cifNet.add(MaxPool2D())
cifNet.add(Dropout(0.2))

cifNet.add(Flatten())
cifNet.add(Dense(128,activation = "relu"))
cifNet.add(Dropout(0.2))
cifNet.add(Dense(42,activation='linear'))

from keras.optimizers import Adam
opt = Adam(lr=0.001)
cifNet.compile(optimizer=opt,
               loss='mean_squared_error',
               metrics=['mse'])

cifNet.fit(trainset[:-300],labelset[:-300],batch_size=128,epochs=15,validation_split=0.1)

model.evaluate(trainset[-300:],labelset[-300:])
### Multiple fits with other datasets give much better results, in fact the loss of the 
### model with transfer learning gets even smaller than 100.

from random import seed
from random import randint

ran = randint(0,len(trainset)-1)
implot = plt.imshow(trainset[ran])
test=np.zeros((1,332,382,3))
test[0] = trainset[ran]
pro_array=model.predict(test)
for x in range(42):
  if x%2==0:
    #plt.scatter([labs[5][x]],[labs[5][x+1]])
    plt.scatter([pro_array[0][x]],[pro_array[0][x+1]],color="red")
# put a blue dot at (10, 20)

