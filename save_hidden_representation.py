import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

flags = tf.app.flags
FLAGS = flags.FLAGS
FLAGS.checkpoint_dir = './models/'
# Define layer to extract

# Define the shapes of x and y without specifying the batch size (None)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Reshape the input vector to images of 28x28
x_image = tf.reshape(x, [-1,28,28,1])

# Define Convolution and Pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Define the parameters of the logistic regression
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#=====================================#
# FIRST LAYER
#=====================================#
# Initialize the weight of the first layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# Define the computation of conv1 and pool1
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#=====================================#
# SECOND LAYER
#=====================================#
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#=====================================#
# FULLY CONNECTED LAYER
#=====================================#
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1_no_relu = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Add dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


#=====================================#
# LAST FULLY CONNECTED LAYER
#=====================================#
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#----------------------------------------------------------#

saver = tf.train.Saver()

l_names = ['h_fc1_no_relu', 'y']
layers = [h_fc1_no_relu, y]
batch_size = 55
iterations = 1000
n_samples = mnist.train.labels.shape[0]
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise Exception('No checkpoint was found')
    for j, (name, layer) in enumerate(zip(l_names, layers)):
        shape = layer.get_shape()[1:]
        hidden_data = np.empty(np.append(batch_size*iterations, shape))
        for i in range(iterations):
            print('[{:05.1f}%] Extracting minibatch {}'.format(
                100.0*(i+1+(j*iterations))*batch_size/(n_samples*len(layers)),i+1))
            batch = mnist.train.next_batch(batch_size)
            hidden_data[i*batch_size:(i+1)*batch_size] = layer.eval(
                    feed_dict={x: batch[0], keep_prob: 1.0})
        np.save('./datasets/mnist_{}'.format(name), hidden_data)
