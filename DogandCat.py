#%%
import os
import shutil
import PIL

#%%

original_dataset_dir="/Users/jessica/CMT309/MLCoursework2/Project1/dogs-vs-cats/train"
base_dir="/Users/jessica/CMT309/MLCoursework2/Project1/dogs-vs-cats_small"

train_dir=os.path.join(base_dir,"train")
validation_dir=os.path.join(base_dir,"validation")
test_dir=os.path.join(base_dir,"test")

train_cats_dir=os.path.join(train_dir,"cats")
train_dogs_dir=os.path.join(train_dir,"dogs")
validation_cats_dir=os.path.join(validation_dir,"cats")
validation__dogs_dir=os.path.join(validation_dir,"dogs")
test_cats_dir=os.path.join(test_dir,"cats")
test_dogs_dir=os.path.join(test_dir,"dogs")
"""
os.mkdir(base_dir)
os.mkdir(train_dir)
os.mkdir(validation_dir)
os.mkdir(test_dir)
os.mkdir(train_cats_dir)
os.mkdir(train_dogs_dir)
os.mkdir(validation__dogs_dir)
os.mkdir(validation_cats_dir)
os.mkdir(test_cats_dir)
os.mkdir(test_dogs_dir)

#%%
fnames=["cat.{}.jpg".format(i) for i in range(1000)]
for fname in fnames:
    src=os.path.join(original_dataset_dir,fname)
    dat=os.path.join(train_cats_dir,fname)
    shutil.copyfile(src,dat)

fnames=["cat.{}.jpg".format(i) for i in range(1000,1500)]
for fname in fnames:
    src=os.path.join(original_dataset_dir,fname)
    dat=os.path.join(validation_cats_dir,fname)
    shutil.copyfile(src,dat)

fnames=["cat.{}.jpg".format(i) for i in range(1500,2000)]
for fname in fnames:
    src=os.path.join(original_dataset_dir,fname)
    dat=os.path.join(test_cats_dir,fname)
    shutil.copyfile(src,dat)

fnames = ["dog.{}.jpg".format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dat = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dat)

fnames = ["dog.{}.jpg".format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dat = os.path.join(validation__dogs_dir, fname)
    shutil.copyfile(src, dat)

fnames = ["dog.{}.jpg".format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dat = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dat)

"""
#%%
##构建网络
from keras import layers
from keras import models

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
model.add(layers.Dense(1,activation="sigmoid"))
model.summary()


#%%
#配置模型
from keras import optimizers
model.compile(loss="binary_crossentropy",optimizer=optimizers.RMSprop(lr=1e-4),metrics=["acc"])

#%%
#图像预处理
from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=20,class_mode="binary")
validation_generator=test_datagen.flow_from_directory(validation_dir,target_size=(150,150),batch_size=20,class_mode="binary")

#%%
history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=30,validation_data=validation_generator,validation_steps=50)

#%%
model.save("cat_and_dog_small_1.h5")



########################################################################

#%%
#画图
import matplotlib.pyplot as plt
acc=history.history["acc"]
val_acc=history.history["val_acc"]
loss=history.history["loss"]
val_loss=history.history["val_loss"]

epochs=range(1,len(acc)+1)
plt.plot(epochs,acc,"bo",label="Training_acc")
plt.plot(epochs,val_acc,"b",label="Validation_acc")
plt.title("Training and Validation accuracy")
plt.legend()

plt.figure()
plt.plot(epochs,loss,"bo",label="Training_loss")
plt.plot(epochs,val_loss,"b",label="Validation_loss")
plt.title("Training and Validation loss")
plt.legend()

plt.show( )

#%%
#添加dropout层
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
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512,activation="relu"))
model.add(layers.Dense(1,activation="sigmoid"))
model.summary()
model.compile(loss="binary_crossentropy",optimizer=optimizers.RMSprop(lr=1e-4),metrics=["acc"])
#数据增强

train_datagen=ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,)
test_datagen=ImageDataGenerator(rescale=1./255)  #不能增强验证集

train_generator=train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=20,class_mode="binary")
validation_generator=test_datagen.flow_from_directory(validation_dir,target_size=(150,150),batch_size=20,class_mode="binary")

history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=100,validation_data=validation_generator,validation_steps=50)
model.save("cat_and_dog_small_2.h5")

#%%
### 添加VGG16模型 ###

from keras.applications import VGG16
import numpy as np
conv_base=VGG16(weights="imagenet",include_top=False,input_shape=(150,150,3))
conv_base.summary() #(150*150*3 -> 4*4*512)

#%%
## 运用VGG16.predict 去提取特征，再塞到密集层中去训练，不需要单独训练VGG16 ##

datagen=ImageDataGenerator(rescale=1./255)
batch_size=20

def extract_features(directory,sample_count):
    features=np.zeros(shape=(sample_count,4,4,512))
    labels=np.zeros(shape=(sample_count))
    generator=datagen.flow_from_directory(directory,target_size=(150,150),batch_size=batch_size,class_mode="binary")
    i=0
    for inputs_batch,labels_batch in generator:
        features_batch=conv_base.predict(inputs_batch) #用predict去提取特征
        features[i*batch_size:(i+1)*batch_size]=features_batch
        labels[i*batch_size:(i+1)*batch_size]=labels_batch
        i+=1
        if i*batch_size>=sample_count:
            break
    return features,labels

train_features,train_labels=extract_features(train_dir,2000)
validation_features,validation_labels=extract_features(validation_dir,1000)
test_features,test_lables=extract_features(test_dir,1000)

#展平
train_features=np.reshape(train_features,(2000,4*4*512))
validation_features=np.reshape(validation_features,(1000,4*4*512))
test_features=np.reshape(test_features,(1000,4*4*512))

#%%
model=models.Sequential()
model.add(layers.Dense(256,activation="relu",input_dim=4*4*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation="sigmoid"))
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),loss="binary_crossentropy",metrics=["acc"])
history=model.fit(train_features,train_labels,epochs=30,batch_size=20,validation_data=(validation_features,validation_labels))

model.save("cat_and_dog_small3.h5")

#%%

model=models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation="relu"))
model.add(layers.Dense(1,activation="sigmoid"))

## 因为要运用VGG16去重新训练，所以要冻结层，保持它权重不变
print(len(model.trainable_weights))
conv_base.trainable=False ##这句话执行之后，conv_base已经改变了。如果要重新执行，需要把conv_base的定义重新修改一下
print(len(model.trainable_weights))

#%%
train_datagen=ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode="nearest")
test_datagen=ImageDataGenerator(rescale=1./255)  #不能增强验证集

train_generator=train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=20,class_mode="binary")
validation_generator=test_datagen.flow_from_directory(validation_dir,target_size=(150,150),batch_size=20,class_mode="binary")
model.compile(loss="binary_crossentropy",optimizer=optimizers.RMSprop(lr=2e-5),metrics=["acc"])
history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=30,validation_data=validation_generator,validation_steps=50)
model.save("cat_and_dog_small_4.h5")

