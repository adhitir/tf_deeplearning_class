#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
sess = tf.InteractiveSession()


# In[2]:


tf.reset_default_graph()

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)


# In[3]:


data.test.cls = np.argmax(data.test.labels, axis=1)
img_size = 28
img_size_flat = img_size * img_size
# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)
num_channels = 1
labels_size = 10


# In[4]:


# Creating placeholders

input = tf.placeholder(tf.float32, [None, img_size*img_size])
labels = tf.placeholder(tf.float32, [None, labels_size])

x_image = tf.reshape(input, [-1, img_size, img_size, num_channels])
label_true_cls = tf.argmax(labels, dimension=1)


# In[5]:


#CNN 1

# layer_conv1
net1 = tf.layers.conv2d(inputs=x_image, name='layer_conv11', padding='same',filters=16, kernel_size=5, activation=tf.nn.relu)
net1 = tf.layers.max_pooling2d(inputs=net1, pool_size=2, strides=2)

# layer_conv2
net1 = tf.layers.conv2d(inputs=net1, name='layer_conv21', padding='same', filters=36, kernel_size=5, activation=tf.nn.relu)
net1 = tf.layers.max_pooling2d(inputs=net1, pool_size=2, strides=2)

net1 = tf.layers.flatten(net1)

net1 = tf.layers.dense(inputs=net1, name='layer_fc11', units=128, activation=tf.nn.relu)
logits1 = tf.layers.dense(inputs=net1, name='layer_fc_out1',  units=labels_size, activation=None)
print(logits1)

label_pred1 = tf.nn.softmax(logits=logits1)
label_pred_cls1 = tf.argmax(label_pred1, dimension=1)

#CNN 2

# layer_conv1
net2 = tf.layers.conv2d(inputs=x_image, name='layer_conv12', padding='same',filters=16, kernel_size=5, activation=tf.nn.relu)
net2 = tf.layers.max_pooling2d(inputs=net2, pool_size=2, strides=2)

# layer_conv2
net2 = tf.layers.conv2d(inputs=net2, name='layer_conv22', padding='same', filters=36, kernel_size=5, activation=tf.nn.relu)
net2 = tf.layers.max_pooling2d(inputs=net2, pool_size=2, strides=2)

# layer_conv3
net2 = tf.layers.conv2d(inputs=net2, name='layer_conv32', padding='same', filters=64, kernel_size=5, activation=tf.nn.relu)
net2 = tf.layers.max_pooling2d(inputs=net2, pool_size=2, strides=2)

net2 = tf.layers.flatten(net2)

net2 = tf.layers.dense(inputs=net2, name='layer_fc12', units=128, activation=tf.nn.relu)
logits2 = tf.layers.dense(inputs=net2, name='layer_fc_out2',  units=labels_size, activation=None)
print(logits2)

label_pred2 = tf.nn.softmax(logits=logits2)
label_pred_cls2 = tf.argmax(label_pred2, dimension=1)

# DNN 1

hidden1 = tf.layers.dense(inputs=input, units=100, activation=tf.nn.relu,name='hidden1') 
hidden2 = tf.layers.dense(inputs=hidden1, units=78, activation=tf.nn.relu,name='hidden2')
hidden3 = tf.layers.dense(inputs=hidden2, units=78, activation=tf.nn.relu,name='hidden3')
hidden4 = tf.layers.dense(inputs=hidden3, units=39, activation=tf.nn.relu,name='hidden4')
hidden5 = tf.layers.dense(inputs=hidden4, units=39, activation=tf.nn.relu,name='hidden5')
logits3 = tf.layers.dense(inputs=hidden5, units=labels_size,name='outputs') 

label_pred3 = tf.nn.softmax(logits=logits3)
label_pred_cls3 = tf.argmax(label_pred3, dimension=1)


# In[6]:


loss1 = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=logits1, labels=labels) )
loss2 = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=logits2, labels=labels) )
xentropy3 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits3)
loss3 = tf.reduce_mean(xentropy3, name="loss3")

optimizer1 = tf.train.AdamOptimizer().minimize(loss1)
optimizer2 = tf.train.AdamOptimizer().minimize(loss2)
optimizer3 = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss3)


correct_prediction1 = tf.equal(label_pred_cls1, label_true_cls)
correct_prediction2 = tf.equal(label_pred_cls2, label_true_cls)
correct_prediction3 = tf.equal(label_pred_cls3, label_true_cls)


accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))
accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3, tf.float32))


# In[8]:


train_batch_size = 100

loss1_list=[]
acc1_list=[]
loss2_list=[]
acc2_list=[]
loss3_list=[]
acc3_list=[]


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    
    total_iterations = data.train.num_examples // train_batch_size

    
    for iteration in range(total_iterations):

        x_batch, label_true_batch = data.train.next_batch(train_batch_size)
        
        
        feed_dict_train = {input: x_batch, labels: label_true_batch}

        _, l1, pred1 = sess.run([optimizer1, loss1, logits1], feed_dict=feed_dict_train)
        _, l2, pred2 = sess.run([optimizer2, loss2, logits2], feed_dict=feed_dict_train)
        _, l3, pred3 = sess.run([optimizer3, loss3, logits3], feed_dict=feed_dict_train)

        los1, acc1 = sess.run([loss1, accuracy1], feed_dict=feed_dict_train)
        los2, acc2 = sess.run([loss2, accuracy2], feed_dict=feed_dict_train)
        los3, acc3 = sess.run([loss3, accuracy3], feed_dict=feed_dict_train)

        loss1_list.append(los1)
        acc1_list.append(acc1)

        loss2_list.append(los2)
        acc2_list.append(acc2)

        loss3_list.append(los3)
        acc3_list.append(acc3)


# In[9]:


fig,ax = fig, ax=plt.subplots()

ax.plot(loss1_list,label='CNN1')
ax.plot(loss2_list,label='CNN2')
ax.plot(loss3_list,label='DNN')


plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
ax.legend();


# In[10]:


fig2,ax2 = fig, ax2=plt.subplots()

ax2.plot(acc1_list,label='CNN1')
ax2.plot(acc2_list,label='CNN2')
ax2.plot(acc3_list,label='DNN')


plt.title('Model accuracy')
plt.ylabel('Loss')
plt.xlabel('Epoch')
ax2.legend();


# In[ ]:




