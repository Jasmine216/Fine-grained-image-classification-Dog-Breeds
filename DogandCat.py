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
#%%
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

#%%
model.compile(loss="binary_crossentropy",optimizer=optimizers.RMSprop(lr=1e-4),metrics=["acc"])

#%%
#图像预处理
from keras.preprocessing.image import ImageDataGenerator

#%%
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
#draw picture to find best parameter

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

##propressing dataset

#添加dropout层 add dropout layer
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
model.add(layers.Dropout(0.5))   ## add dropout layer
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

##build model

### 添加VGG16模型 ###

from keras.applications import VGG16
import numpy as np
conv_base=VGG16(weights="imagenet",include_top=False,input_shape=(150,150,3))
conv_base.summary() #(150*150*3 -> 4*4*512)

#%%
## 第一种办法：运用VGG16.predict 去提取特征，再塞到密集层中去训练，不需要单独训练图片 ##

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
##第二种办法：把VGG16的模型添加到其中进行训练+数据增强
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

#%%
# model.compile(loss="binary_crossentropy",optimizer=optimizers.RMSprop(lr=2e-5),metrics=["acc"])
history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=30,validation_data=validation_generator,validation_steps=50)
model.save("cat_and_dog_small_4.h5")


#%%
#第三种办法：微调模型,只调下层（第五层）
conv_base.trainable=True

set_trainable=False
for layer in conv_base.layers:
    if layer.name=='block5_conv1':
        set_trainable=True
    if set_trainable:
        layer.trainable=True
    else:
        layer.trainable=False

#%%
model.compile(loss="binary_crossentropy",optimizer=optimizers.RMSprop(lr=1e-5),metrics=["acc"])
history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=100,validation_data=validation_generator,validation_steps=50)

#%%
#从测试数据集上最终评估这个模型
test_generator=test_datagen.flow_from_directory(
    test_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode="binary"
)
#%%
#第一次出现验证环节
test_loss,test_acc=model.evaluate_generator(test_generator,steps=50)
print("test acc:",test_acc)


#%%
#开始可视化:

# 第一种： 可视化中间激活

from keras.models import load_model
model=load_model("cat_and_dog_small_2.h5")
model.summary()

#%%
#预处理单个图像
img_path="/Users/jessica/CMT309/MLCoursework2/Project1/dogs-vs-cats_small/test/cats/cat.1500.jpg"
from keras.preprocessing import image
import numpy as np
img=image.load_img(img_path,target_size=(150,150))
img_tensor=image.img_to_array(img) #(150,150,3)
img_tensor=np.expand_dims(img_tensor,axis=0) #(1,150,150,3) 因为模型第一层是四位
img_tensor/=225.
#%%
print(img_tensor[0].shape)

#%%
#显示图像
import matplotlib.pyplot as plt
plt.imshow(img_tensor[0]) #就是本来的（150，150，3）
plt.show()

#%%
#构建一个激活模型，并且预测出每层的结果
from keras import models
layer_outputs=[layer.output for layer in model.layers]
activation_model=models.Model(input= model.input,output=layer_outputs)
#平时模型是一个输入，一个输出；这个模型是一个输出，八个输出
#print(model.input)

activation=activation_model.predict(img_tensor)#返回一个八位列表，每一位对应一个层对应一个Numpy数组
first_layer_activation=activation[0]
print(first_layer_activation.shape) #第0层出来的结果 （1，148，148，32）

#%%
#显示第0层的第几个通道的图
import matplotlib.pyplot as plt
plt.matshow(first_layer_activation[0,:,:,6],cmap="viridis")
plt.show()


#%%
# 第二种： 可视化卷积神经网络的过滤器

from keras.applications import VGG16
from keras import backend as K

#%%
model=VGG16(weights="imagenet",include_top=False)
# https://blog.csdn.net/sunshunli/article/details/81431500 about VGG16
layer_name="block3_conv1"
filter_index=0
layer_output=model.get_layer(layer_name).output
#%%
loss=K.mean(layer_output[:,:,:,filter_index]) #损失函数
grads=K.gradients(loss,model.input)[0] #这个损失相对于输入图像的梯度,model.output(loss)对model.input(输入图像)的求导
grads/=(K.sqrt(K.mean(K.square(grads)))+1e-5) #梯度标准化
iterate=K.function([model.input],[loss,grads]) #定一个一个函n数：输入函数，得到loss和grads
step=1
input_img_data=""
for i in range(40):
    loss_value,grads_value=iterate([input_img_data])
    input_img_data*=grads_value*step #当前图*重要性

#%%
#第三种： 可视化类激活的热力图

from keras.applications.vgg16 import VGG16
import numpy as np
model=VGG16(weights="imagenet")
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input,decode_predictions
img_path="/Users/jessica/CMT309/MLCoursework2/Project1/dogs-vs-cats_small/test/cats/cat.1500.jpg"
img=image.load_img(img_path,target_size=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
x=preprocess_input(x) #这些训练图像都根据keras.applications.vgg16.preprocess_input函数中的内置的规则进行预处理
preds=model.predict(x)
print(decode_predictions(preds,top=3))
#%%
#decode_predictions 输出3个最高概率：(类名, 语义概念, 预测概率)
africa_elephant_output=model.output[:,386]
last_conv_layer=model.get_layer("block5_conv3")
#print(africa_elephant_output.shape)
#print(last_conv_layer.output.shape)
grads=K.gradients(africa_elephant_output,last_conv_layer.output)[0] #梯度是对变量的"导数和"
print(grads.shape)

#%%
import matplotlib.pyplot as plt
pooled_grads=K.mean(grads,axis=(0,1,2))#形状为（512，）的向量。每个元素是特定特征通道图的平均梯度大小
iterate=K.function([model.input],[pooled_grads,last_conv_layer.output[0]])
pooled_grads_value,last_conv_layer_value=iterate([x]) # 得到"梯度"和"输出层"
for i in range(512):
    last_conv_layer_value[:,:,i]*=pooled_grads_value[i] # 一共512个通道，每个通道*这个通道对于"大象"类别的重要程度

hearmap=np.mean(last_conv_layer_value,axis=-1) #得到逐通道平均值为类激活的热力图
hearmap=np.maximum(hearmap,0) #负的舍弃为0
hearmap/=np.max(hearmap) #除以最大值化为0-1的值
plt.matshow(hearmap)
plt.show()

#%%
#将热力图与原始图像叠加

import cv2
img=cv2.imread(img_path)
hearmap=cv2.resize(hearmap,(img.shape[1],img.shape[0]))
hearmap=np.unit8(255*hearmap) #转化为RGB格式
hearmap=cv2.applyColorMap(hearmap,cv2.COLORMAP_JET)  #将热力图应用于原始图像
superimposed_img=hearmap*0.4+img  #这里的0.4是热力图强度因子
cv2.imshow(superimposed_img)


