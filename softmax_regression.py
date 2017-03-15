#Softmax Regression model
#Loading dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
#import library of TensorFlow
import tensorflow as tf
#Create a new InteractiveSession for computing. Sessions are independent with each other.
sess = tf.InteractiveSession()
#Create a new Placeholder to store data with two parameter, data type and shape.
#Here, "None" represent it's no limted amount to inputs. "784" is the dimension of inputs.


x = tf.placeholder(tf.float32, [None, 784])
#"W" is the weights and "b" represents  bias, and both of them are initized by zero.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
#The formula of Softmax Regression.
#"tf.nn" includes many functions of nerual network, and "tf.matmul" is the matrix multiplication.
y = tf.nn.softmax(tf.matmul(x, W) + b)
#Defining loss function by cross entropy.
#"tf.reduce_mean" is to compute the average and "tf.reduce_sum" for summation.
y_ = tf.placeholder(tf.float32, [None, 10])


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))
#Learning rate: 0.5. Optimation goal: minimization of cross entropy.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


tf.global_variables_initializer().run()


#Training: random gradient descent by mini-batch of 100 random samples.
#Times of learning: 1000.
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	train_step.run({x: batch_xs, y_: batch_ys})


#The accracy of Softmax Regression is about 92%.
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accracy.eval({x: mnist.test.images, y_: mnist.test.labels}))