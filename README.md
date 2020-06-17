# dog recognition

# Preprocessing
* Split the dataset into sets at 8:1:1 by circular copy and external library respectively but this part of codes are discarded in the end because of preprocessing of cropped images from other team members.
 
* Make three generators for three image data sets to convert images to floating arrays.

# Create the initial model
* Create the initial model with convolutional layers, max-pooling layers and dense layers. However, this model results in low accuracy.
 
*  Communicate with the team member who used VGG16 to train the data and got a higher accuracy. Then I started to try the Xception model and discarded the self-created model created in the beginning. Compared with VGG16, the Xception model results in higher accuracy. 
	 
# Solve the problem of overfitting
I started from the literature review and found ways to solve the problem of overfitting. Three ways are tried to solve the problem:
 
## Data augmentation 
Data augmentation is to solve the problem of overfitting by increasing the number of training images. The model will not view the same image twice during training. This allows the model to observe more data and thus have better generalization capabilities.

We generate augmented data before training the classifier. ImageDataGenerator() is used to make basic transformations for all images in the training set. All these images are fed into the net at training time. (Perez and Wang, 2020) Figure 1 shows four types of transformations for one image. 

![Figure1](https://github.com/Jasmine216/Fine-grained-image-classification-Dog-Breeds/blob/pictures/image.png)

Data Augmentation results in the decrease of the accuracy of the training set but helps solve overfitting and improve the accuracy of developments set on a small scale. (Table 1) 

![Table 1](https://github.com/Jasmine216/Fine-grained-image-classification-Dog-Breeds/blob/pictures/table1.png)

## Dropout 
Dropout is the most effective way of regularization to solve overfitting. It can discard some features randomly of the upper layer. The dropout rate is set between 0.2 and 0.5. We set it 0.3 after a Dense layer in the model. This way decreases the accuracy of the training set but dropout() is useless for development and testing sets. 
## Feature extraction 
We use the convolutional base of the model to train a new classifier. The convolutional base can obtain the feature of the objective location. The generality (and reusability) of the representation extracted by a convolutional layer depends on the depth of the layer in the model. The layer closer to the bottom in the model extracts local, highly versatile feature maps (such as visual edges, colors, and textures), while the layer closer to the top extracts more abstract concepts (such as "dog ears" or "dog eyes"). Thus, except for the convolutional base from the Xception, we also try the convolutional base from the InceptionResnet who has the deepest layers. 
* Two ways to extract features: 
	* 1  Feature extraction without data augmentation 
	* 2  Feature extraction with data augmentation 

If the features are extracted without data augmentation, the training efficiency can be increased compared with feature extraction with data augmentation. More time is used in feature extraction than fitting the model. If we pay more attention to the speed of improving the model result, we can apply it. 
* Two kinds of convolutional bases to extract features: 
	* 1  Convolutional base from the Xception 
	* 2  Convolutional base from the InceptionResnet 

Feature extraction helps solve the problem of overfitting and decreases the accuracy of the training set. It helps increase the accuracy of the development set. We can see that extracting features by the convolutional base of InceptionResnet with data augmentation and dropout can result in the highest accuracy. (Table 2) 

![Table2](https://github.com/Jasmine216/Fine-grained-image-classification-Dog-Breeds/blob/pictures/table2.png)

## Tune parameters for the model
### Class mode and Loss function 
We choose the sparse categorical cross-entropy as the loss function because we have multiple classes and each sample belongs exactly to one class. We chose the categorical cross-entropy as the loss function by mistake in the beginning, which resulted in the low accuracy. Then we find categorical cross-entropy is chosen when one sample belongs to multiple classes. The class mode“sparse" is used here to get integer labels, also matching the loss function.  
### Optimizer comparison 
Common choices for the optimizer are either a fixed value, a predetermined schedule, or an adaptive scheme based on curvature or gradient information (Keskar and Saon, 2015). SGD and Adam are the main optimizers compared in the model. 
SGD is our first choice of the optimizer with default parameters as an attempt. Surprisingly, it results in a good result. However, the accuracy fluctuates around one value because of the fixed but too large learning rate. Thus, the callback function of reducing the learning rate when the accuracy doesn't change is added. Then the accuracy is improved on a small scale.
Adam is an adaptive scheme based on momentum, which is stronger than SGD. The Parameters ß1 is used in the first-order momentum and ß2 is used in the second-order momentum. The problem of choosing parameters has been investigated by the optimization community with many options available which ensure global convergence under regularity conditions. The recommended parameters: 
optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), metrics = ["accuracy"]) 
We can get the result that the optimizer SGD with the callback function of reducing the learning rate can get higher accuracy. (Table 3) 

### Batch size 
The choice of batch size should be between 2 and 32m because mini-batch can update parameters many times in one epoch to speed up convergence. (Masters and Luschi, 2020) Another reason is that we used the batch of data instead of the full data can Escaping From Saddle Points. (Ge, Huang, Jin and Yuan, 2020) We set batch size 20 here. 

![Table3](https://github.com/Jasmine216/Fine-grained-image-classification-Dog-Breeds/blob/pictures/table3.png)
 
# Conclusion and Improvement

* In all, my ways to solve the problem of overfitting and tune parameters worked. The accuracy of the model increased to 93%.

![](https://github.com/Jasmine216/Fine-grained-image-classification-Dog-Breeds/blob/pictures/result.png)
 
* Some changes I will make to improve next time. Firstly, more communication with members before creating the model. The way of preprocessing of data sets was changed after my work of tuning parameters, which wastes more time to readjust the parameters. Besides, the location of images and cropping of images can improve the results dramatically. Thus, I should pay more attention to the preprocessing of data sets instead of tuning parameters of the model next time.
 
 
# Reference
* Perez, L., and Wang, J. 2020. The Effectiveness Of Data Augmentation In Image Classification Using Deep Learning. [online] arXiv.org. Available at: <https://arxiv.org/abs/1712.04621> [Accessed 19 April 2020]. 
* Keskar, N, S., and Saon, G., 2015. A non-monotone learning rate strategy for SGD training of deep neural networks. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Brisbane, QLD, 2015, pp. 4974-4978.
* Masters, D. and Luschi, C., 2020. Revisiting Small Batch Training For Deep Neural Networks. [online] arXiv.org. Available at: <https://arxiv.org/abs/1804.07612> [Accessed 20 April 2020]. 
* Ge, R., Huang, F., Jin, C. and Yuan, Y., 2020. Escaping From Saddle Points --- Online Stochastic Gradient For Tensor Decomposition. [online] arXiv.org. Available at: <https://arxiv.org/abs/1503.02101> [Accessed 20 April 2020]. 
