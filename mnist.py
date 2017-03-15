#MNIST: Mixed National Institute of Standards and Technology database
#The one of simple datasets for machine learning
#MNIST includes tens of thousands of handwriting pictuer with scale by 28px × 28px
#Tensorflow has a package to load MNIST data.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

#The dataset includes train set (55000 samples), validation set (5000 samples) and test set (10000 samples)
#Each sample has its label.
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)
