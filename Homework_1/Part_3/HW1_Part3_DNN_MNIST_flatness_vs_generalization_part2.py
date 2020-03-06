#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
tf.logging.set_verbosity(tf.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}
tf.reset_default_graph()


from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)
data.test.cls = np.argmax(data.test.labels, axis=1)


# In[2]:


img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10


# In[3]:


def get_weights_variable(layer_name):
    # Retrieve an existing variable named 'kernel' in the scope with the given layer_name.

    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('kernel',dtype=tf.float32)

    return variable


# In[4]:


def get_bias_variable(layer_name):
    # Retrieve an existing variable named 'kernel' in the scope with the given layer_name.

    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('bias',dtype=tf.float32)

    return variable


# In[5]:


# Train the model 


# In[6]:


n_epochs = 10

batch_size_arr = [10,20,40,60,80,100,200,400,600,800,1000]

total_parameters = 0

tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
y_true = tf.placeholder(tf.float32, shape=[None,num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

# # Model 1
# #########

# hidden1 = tf.layers.dense(inputs=x, units=100, activation=tf.nn.relu,name='hidden1') 
# hidden2 = tf.layers.dense(inputs=hidden1, units=78, activation=tf.nn.relu,name='hidden2')
# hidden3 = tf.layers.dense(inputs=hidden2, units=78, activation=tf.nn.relu,name='hidden3')
# logits = tf.layers.dense(inputs=hidden3, units=num_classes,name='outputs') 

# y_pred = tf.nn.softmax(logits=logits)
# y_pred_cls = tf.argmax(y_pred, dimension=1)

# xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
# loss = tf.reduce_mean(xentropy, name="loss")
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# correct = tf.equal(y_pred_cls, y_true_cls) # correct is a boolean array of size #batch_size#
# accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# w_out = get_weights_variable(layer_name='outputs')
# grads = tf.gradients(loss, x)[0]
# #hessian = tf.reduce_sum(tf.hessians(loss, x)[0], axis = 2)
# ########

# # Model 2
# #########

# hidden1 = tf.layers.dense(inputs=x, units=100, activation=tf.nn.relu,name='hidden1') 
# hidden2 = tf.layers.dense(inputs=hidden1, units=78, activation=tf.nn.relu,name='hidden2')
# hidden3 = tf.layers.dense(inputs=hidden2, units=78, activation=tf.nn.relu,name='hidden3')
# hidden4 = tf.layers.dense(inputs=hidden3, units=39, activation=tf.nn.relu,name='hidden4')
# hidden5 = tf.layers.dense(inputs=hidden4, units=39, activation=tf.nn.relu,name='hidden5')
# logits = tf.layers.dense(inputs=hidden5, units=num_classes,name='outputs') 

# y_pred = tf.nn.softmax(logits=logits)
# y_pred_cls = tf.argmax(y_pred, dimension=1)

# xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
# loss = tf.reduce_mean(xentropy, name="loss")
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# correct = tf.equal(y_pred_cls, y_true_cls) # correct is a boolean array of size #batch_size#
# accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# w_out = get_weights_variable(layer_name='outputs')
# grads = tf.gradients(loss, x)[0]

# ########

# Model 3
#########

# # layer_conv1
# net = tf.layers.conv2d(inputs=x_image, name='layer_conv11', padding='same',filters=16, kernel_size=5, activation=tf.nn.relu)
# net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

# # layer_conv2
# net = tf.layers.conv2d(inputs=net, name='layer_conv21', padding='same', filters=36, kernel_size=5, activation=tf.nn.relu)
# net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

# net = tf.layers.flatten(net)

# net = tf.layers.dense(inputs=net, name='layer_fc11', units=128, activation=tf.nn.relu)
# logits = tf.layers.dense(inputs=net, name='layer_fc_out1',  units=num_classes, activation=None)

# y_pred = tf.nn.softmax(logits=logits)
# y_pred_cls = tf.argmax(y_pred, dimension=1)

# xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
# loss = tf.reduce_mean(xentropy, name="loss")
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# correct = tf.equal(y_pred_cls, y_true_cls) # correct is a boolean array of size #batch_size#
# accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# #w_out = get_weights_variable(layer_name='layer_fc_out1')
# grads = tf.gradients(loss, x)[0]
# # #hessian = tf.reduce_sum(tf.hessians(loss, x)[0], axis = 2)
# ########

# # Model 4
# #########

# layer_conv1
net = tf.layers.conv2d(inputs=x_image, name='layer_conv1', padding='same',filters=36, kernel_size=5, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

# layer_conv2
net = tf.layers.conv2d(inputs=net, name='layer_conv2', padding='same', filters=16, kernel_size=5, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

# layer_conv3
net = tf.layers.conv2d(inputs=net, name='layer_conv3', padding='same', filters=8, kernel_size=5, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

# layer_conv4
net = tf.layers.conv2d(inputs=net, name='layer_conv4', padding='same', filters=8, kernel_size=5, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

net = tf.layers.flatten(net)

net = tf.layers.dense(inputs=net, name='layer_fc1', units=128, activation=tf.nn.relu)
logits = tf.layers.dense(inputs=net, name='layer_fc_out',  units=num_classes, activation=None)

y_pred = tf.nn.softmax(logits=logits)
y_pred_cls = tf.argmax(y_pred, dimension=1)

xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
loss = tf.reduce_mean(xentropy, name="loss")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

correct = tf.equal(y_pred_cls, y_true_cls) # correct is a boolean array of size #batch_size#
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

#w_out = get_weights_variable(layer_name='layer_fc_out1')
grads = tf.gradients(loss, x)[0]
# #hessian = tf.reduce_sum(tf.hessians(loss, x)[0], axis = 2)

# ########

# # Model 5
# #########

# layer_conv1
net = tf.layers.conv2d(inputs=x_image, name='layer_conv1', padding='same',filters=64, kernel_size=10, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

# layer_conv2
net = tf.layers.conv2d(inputs=net, name='layer_conv2', padding='same', filters=32, kernel_size=5, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

# layer_conv3
net = tf.layers.conv2d(inputs=net, name='layer_conv3', padding='same', filters=16, kernel_size=5, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

# layer_conv4
net = tf.layers.conv2d(inputs=net, name='layer_conv4', padding='same', filters=8, kernel_size=5, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

net = tf.layers.flatten(net)

net = tf.layers.dense(inputs=net, name='layer_fc1', units=128, activation=tf.nn.relu)
logits = tf.layers.dense(inputs=net, name='layer_fc_out',  units=num_classes, activation=None)

y_pred = tf.nn.softmax(logits=logits)
y_pred_cls = tf.argmax(y_pred, dimension=1)

xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
loss = tf.reduce_mean(xentropy, name="loss")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

correct = tf.equal(y_pred_cls, y_true_cls) # correct is a boolean array of size #batch_size#
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

#w_out = get_weights_variable(layer_name='layer_fc_out1')
grads = tf.gradients(loss, x)[0]
# #hessian = tf.reduce_sum(tf.hessians(loss, x)[0], axis = 2)

# ########


loss_list_train=[]
acc_list_train=[]
loss_list_test=[]
acc_list_test=[]
sensitivity = []

weight_full = []

for batch_size in batch_size_arr:

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())     

        for epoch in range(n_epochs):

            total_iterations = data.train.num_examples // batch_size

            for iteration in range(total_iterations):

                X_batch, y_batch = data.train.next_batch(batch_size)
                feed_dict_train = {x: X_batch, y_true: y_batch}

                _, loss_train, pred = sess.run([optimizer, loss, logits], feed_dict=feed_dict_train)
                los, acc_train = sess.run([loss, accuracy], feed_dict=feed_dict_train)
                grad_batch = sess.run([grads], feed_dict=feed_dict_train)

                grad_reshape = np.array(grad_batch)
                grad_reshape = grad_reshape.reshape(batch_size,784)
                fro_norm = np.linalg.norm(grad_reshape,ord='fro')
                

        total_iterations = data.test.num_examples // batch_size

        for iteration in range(total_iterations):

            X_batch, y_batch = data.test.next_batch(batch_size)
            feed_dict_test = {x: X_batch, y_true: y_batch}

            loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_test)
            grad_batch = sess.run([grads], feed_dict=feed_dict_test)

            grad_reshape = np.array(grad_batch)
            grad_reshape = grad_reshape.reshape(batch_size,784)
            fro_norm = np.linalg.norm(grad_reshape,ord='fro')

                #print(grad_batch)
        sensitivity.append(fro_norm)
        
        loss_list_train.append(loss_train)
        loss_list_test.append(loss_test)

        acc_list_train.append(acc_train)
        acc_list_test.append(acc_test)


        batch_size = batch_size+10


# In[ ]:





# In[7]:


import matplotlib 
matplotlib.rcParams['figure.figsize'] = [8, 6]

fig,ax = fig, ax=plt.subplots()
               
ax.plot(batch_size_arr,loss_list_train,label='Training Loss')
ax.plot(batch_size_arr,loss_list_test,label='Test Loss')
ax.plot(batch_size_arr,sensitivity,label='Sensitivity')

ax.set_xscale('log')
ax.set_title('Generalizations')
ax.set_ylabel('Loss')
ax.set_xlabel('Batch Size')
ax.legend();


# In[8]:


import matplotlib 
matplotlib.rcParams['figure.figsize'] = [8, 6]

fig,ax = fig, ax=plt.subplots()
               
ax.plot(batch_size_arr,acc_list_train,label='Training Accuracy')
ax.plot(batch_size_arr,acc_list_test,label='Test Accuracy')
ax.plot(batch_size_arr,sensitivity,label='Sensitivity')

ax.set_xscale('log')
ax.set_title('Generalizations')
ax.set_ylabel('Accuracy')
ax.set_xlabel('Batch Size')
ax.legend();


# In[9]:


tf.trainable_variables()

