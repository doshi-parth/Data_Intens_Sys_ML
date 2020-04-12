#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

input_dimensions = 28
stride_size = 2
no_of_classes = 10
total_dimensions = 784
conv_size = 5
keep_prob = tf.placeholder(tf.float32)

### All the weights and bias vectors are defined using these 
### helper functions.
def weightVector(shape):
    initial = tf.truncated_normal(shape, mean=0.0, stddev=0.1)
    return tf.Variable(initial)

def biasVector(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

### The convolutional neural network layers are defined using the 
### following two conv2d and maxPool2d functions
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxPool2d(x, stride_size):
    return tf.nn.max_pool(x, ksize=[1, stride_size, stride_size, 1], strides=[1, stride_size, stride_size, 1], padding='SAME')

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, [None, total_dimensions])
x_image = tf.reshape(x, [-1, input_dimensions, input_dimensions, 1])


# In[2]:


W_conv1 = weightVector([conv_size, conv_size, 1, 32])
b_conv1 = biasVector([32])
W_conv2 = weightVector([conv_size, conv_size, 32, 64])
b_conv2 = biasVector([64])
W_fc1 = weightVector([7 * 7 * 64, 1024])
b_fc1 = biasVector([1024])
W_fc2 = weightVector([1024, no_of_classes])
b_fc2 = biasVector([no_of_classes])


# In[3]:


conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
conv1 = maxPool2d(conv1, stride_size)
conv2 = tf.nn.relu(conv2d(conv1, W_conv2) + b_conv2)
conv2 = maxPool2d(conv2, stride_size)
conv2 = tf.reshape(conv2, [-1, 7 * 7 * 64])
fc1 = tf.nn.relu(tf.matmul(conv2, W_fc1) + b_fc1)
fc1_drop = tf.nn.dropout(fc1, keep_prob)
y = tf.nn.softmax(tf.matmul(fc1_drop, W_fc2) + b_fc2)


# In[4]:


y_label = tf.placeholder(tf.float32, [None, no_of_classes])
entropy_loss = -tf.reduce_sum(y_label * tf.log(y))
train = tf.train.AdamOptimizer(1e-4).minimize(entropy_loss)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[5]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[ ]:

writer = tf.summary.FileWriter('./graphs/a1b', tf.get_default_graph())
for step in range(3000):
    batch_x, batch_y = mnist.train.next_batch(50)
    
    if step % 100 == 0:
        loss, acc = sess.run([entropy_loss,accuracy], feed_dict={
                x: batch_x,
                y_label: batch_y,
                keep_prob: 1.0
            })
        print (loss)
        print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))
    
    sess.run(train, feed_dict={
            x: batch_x,
            y_label: batch_y,
            keep_prob: 0.5
        })


# In[ ]:


print("Testing Accuracy:",         sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_label: mnist.test.labels,
                                      keep_prob:1.0}))
writer.close()
