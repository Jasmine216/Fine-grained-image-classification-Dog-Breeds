#%%
import scipy.io
import scipy.io
import numpy as np
import pandas as pd

##load matlab and make datafram to save label and file_id
path= '/Users/jessica/CMT309/MLCoursework2/lists/file_list.mat'
data = scipy.io.loadmat(path)
#%%
#data.keys()
#data.get("__header__") #null
#data.get("__version__") #null
#data.get("__globals__") #null
#data.get("file_list")
#data.get("annotation_list")
#data.get("label") #null
#%%
df = pd.DataFrame()
for i in data:
  if '__' not in i and 'readme' not in i:
    data_array = data[i]
    if df.empty:
      df = pd.DataFrame(data_array)
print(df.head())

#%%
full_list = pd.DataFrame(columns=['id','label'])
for i in range(len(df)):
    full_list.loc[i,'label']=str(str(df.loc[i,0]).split('/')[0]).split('-')[1] #label
    full_list.loc[i, 'id'] = str(str(df.loc[i, 0]).split("/")[1]).split("'")[0] # file name
#%%
full_list.head()

##create path and save all images
#%%
import os
import shutil
original_dataset_dir="/Users/jessica/CMT309/MLCoursework2/Images"
base_dir="/Users/jessica/CMT309/MLCoursework2/Images_New_Path"

#create train directory, validation_directory, test_directory

train_dir=os.path.join(base_dir,"train")
validation_dir=os.path.join(base_dir,"validation")
test_dir=os.path.join(base_dir,"test")
if not os.path.exists(base_dir):
    os.mkdir(base_dir)
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

#%%
#copy 80% images of all folders to train directory and change names
#copy 10% images of all folders to validation...
#copy 10% images of all folders to test....
for i in os.listdir(original_dataset_dir):
    ori_path=os.path.join(original_dataset_dir,i)
    trainpath=os.path.join(train_dir,i)
    if not os.path.exists(trainpath):
        os.mkdir(trainpath)
    validpath = os.path.join(validation_dir, i)
    if not os.path.exists(validpath):
        os.mkdir(validpath)
    testpath=os.path.join(test_dir,i)
    if not os.path.exists(testpath):
        os.mkdir(testpath)
    #print(i)
    type=i.split("-")[0]
    #print(type)
    name=i.split("-")[1]
    #print(int(len(ori_path)*0.8),int(len(ori_path)*0.9),len(ori_path))
    #train_data
    for fname in os.listdir((ori_path))[:int(len(ori_path)*0.8)]:
        #print(fname)
        id=fname.split("_")[-1]
        #print(id)
        src=os.path.join(ori_path,fname)
        new_name=os.path.join(ori_path,name+'_'+id)
        dst = os.path.join(trainpath, name+'_'+id)
        #print(src)
        if os.path.exists(src):
            print("copy1")
            os.rename(src,new_name)
            shutil.copyfile(new_name,dst)
    #validation_data
    for fname in os.listdir((ori_path))[int(len(ori_path)*0.8):int(len(ori_path)*0.9)]:
        #print(fname)
        id=fname.split("_")[-1]
        #print(id)
        src=os.path.join(ori_path,fname)
        new_name=os.path.join(ori_path,name+'_'+id)
        dst = os.path.join(validpath, name+'_'+id)
        #print(src)
        if os.path.exists(src):
            print("copy2")
            os.rename(src,new_name)
            shutil.copyfile(new_name,dst)
    #test_data
    for fname in os.listdir((ori_path))[int(len(ori_path)*0.9):]:
        #print(fname)
        id=fname.split("_")[-1]
        #print(id)
        src=os.path.join(ori_path,fname)
        new_name=os.path.join(ori_path,name+'_'+id)
        dst = os.path.join(testpath, name+'_'+id)
        #print(src)
        if os.path.exists(src):
            print("copy3")
            os.rename(src,new_name)
            shutil.copyfile(new_name,dst)

#%%
#proprecessing of images
from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)
train_generator=train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=20,class_mode="categorical")
validation_generator=test_datagen.flow_from_directory(validation_dir,target_size=(150,150),batch_size=20,class_mode="categorical")

#%%
from keras import layers
from keras import models
import PIL
#%%

#create model
model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(150,150,3))) #(148,148,32)
model.add(layers.MaxPool2D((2,2))) #(148/2=74,74,32)
model.add(layers.Conv2D(64,(3,3),activation="relu")) #(72,72,64)
model.add(layers.MaxPool2D(2,2)) #(36,36,64)
model.add(layers.Conv2D(128,(3,3),activation="relu")) #(34,34,128)
model.add(layers.MaxPool2D(2,2)) #(17,17,128)
model.add(layers.Conv2D(128,(3,3),activation="relu")) #(15,15,128)
model.add(layers.MaxPool2D(2,2)) #(7,7,128)
model.add(layers.Flatten()) #7*7*128=6272
model.add(layers.Dense(512,activation="relu"))
model.add(layers.Dense(120,activation="sigmoid"))
model.summary()
#%%
#fit model
from keras import optimizers
model.compile(loss="categorical_crossentropy",optimizer=optimizers.RMSprop(lr=1e-4),metrics=["acc"])
history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=30,validation_data=validation_generator,validation_steps=50)
