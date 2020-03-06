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


def get_no_of_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print(total_parameters)
    return total_parameters


# In[5]:


n_epochs = 10
batch_size = 2000

trainings = 10

loss_list_train=[]
acc_list_train=[]
loss_list_test=[]
acc_list_test=[]
params_list=[]

unit1 = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
#unit1 = [5, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500] 
unit2 = [5, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500] 
unit3 = [5, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500] 


for i in range(trainings):
        
    tf.reset_default_graph()
       
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
    y_true = tf.placeholder(tf.float32, shape=[None,num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)
    
    hidden1 = tf.layers.dense(inputs=x, units=unit1[i], activation=tf.nn.relu,name='hidden1') 
    hidden2 = tf.layers.dense(inputs=hidden1, units=unit2[i], activation=tf.nn.relu,name='hidden2')
    hidden3 = tf.layers.dense(inputs=hidden2, units=unit3[i], activation=tf.nn.relu,name='hidden3')

    logits = tf.layers.dense(inputs=hidden3, units=num_classes,name='outputs') 
    
    params = get_no_of_parameters()
    params_list.append(params)

    y_pred = tf.nn.softmax(logits=logits)
    y_pred_cls = tf.argmax(y_pred, dimension=1)
    
    xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    print(hidden1)
    print(hidden2)
    print(logits)
    
    correct = tf.equal(y_pred_cls, y_true_cls) # correct is a boolean array of size #batch_size#
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())     

        for epoch in range(n_epochs):

            total_iterations = data.train.num_examples // batch_size
            
            for iteration in range(total_iterations):

                X_train, y_train = data.train.next_batch(batch_size)

                feed_dict_train = {x: X_train, y_true: y_train}

                _, l, pred = sess.run([optimizer, loss, logits], feed_dict=feed_dict_train)
                
                loss_train, acc_train = sess.run([loss, accuracy], feed_dict=feed_dict_train)
                             
                    
        #test
                

        feed_dict_test = {x: data.test.images , y_true: data.test.labels}

        loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_test)


        loss_list_train.append(loss_train)
        loss_list_test.append(loss_test)

        acc_list_train.append(acc_train)
        acc_list_test.append(acc_test)


# In[6]:


fig1,ax1 = fig1, ax1=plt.subplots()

ax1.scatter(params_list, loss_list_train,label='Training Loss')
ax1.scatter(params_list, loss_list_test,label='Testing Loss')

plt.ylabel('Loss')
plt.xlabel('Iteration')
ax1.legend();

fig2,ax2 = fig2, ax2=plt.subplots()

ax2.scatter(params_list, acc_list_train,label='Training Accuracy')
ax2.scatter(params_list, acc_list_test,label='Testing Accuracy')

plt.ylabel('grad')
plt.xlabel('Iteration')
ax2.legend();

