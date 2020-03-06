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


#This code shows the interpolation between two models with different batch sizes.


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

batch_size_m1 = 50
batch_size_m2 = 1000

total_parameters = 0


tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
y_true = tf.placeholder(tf.float32, shape=[None,num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


# Model 1
#########
hidden11 = tf.layers.dense(inputs=x, units=100, activation=tf.nn.relu,name='hidden11') 
hidden21 = tf.layers.dense(inputs=hidden11, units=78, activation=tf.nn.relu,name='hidden21')
hidden31 = tf.layers.dense(inputs=hidden21, units=78, activation=tf.nn.relu,name='hidden31')

logits1 = tf.layers.dense(inputs=hidden31, units=num_classes,name='outputs1') 

y_pred1 = tf.nn.softmax(logits=logits1)
y_pred_cls1 = tf.argmax(y_pred1, dimension=1)

xentropy1 = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits1)
loss1 = tf.reduce_mean(xentropy1, name="loss1")
optimizer1 = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss1)

correct1 = tf.equal(y_pred_cls1, y_true_cls) # correct is a boolean array of size #batch_size#
accuracy1 = tf.reduce_mean(tf.cast(correct1, tf.float32))
########


# Model 2
#########
hidden12 = tf.layers.dense(inputs=x, units=100, activation=tf.nn.relu,name='hidden12') 
hidden22 = tf.layers.dense(inputs=hidden12, units=78, activation=tf.nn.relu,name='hidden22')
hidden32 = tf.layers.dense(inputs=hidden22, units=78, activation=tf.nn.relu,name='hidden32')

logits2 = tf.layers.dense(inputs=hidden32, units=num_classes,name='outputs2') 

y_pred2 = tf.nn.softmax(logits=logits2)
y_pred_cls2 = tf.argmax(y_pred2, dimension=1)

xentropy2 = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits2)
loss2 = tf.reduce_mean(xentropy2, name="loss2")
optimizer2 = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss2)

correct2 = tf.equal(y_pred_cls2, y_true_cls) # correct is a boolean array of size #batch_size#
accuracy2 = tf.reduce_mean(tf.cast(correct2, tf.float32))
########


loss_list_m1=[]
acc_list_m1=[]

loss_list_m2=[]
acc_list_m2=[]

weight_full_m1 = []
weight_full_m2 = []



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())     

    for epoch in range(n_epochs):

        total_iterations_m1 = data.train.num_examples // batch_size_m1
        total_iterations_m2 = data.train.num_examples // batch_size_m2
        
        for iteration in range(total_iterations_m1):

            X_batch_m1, y_batch_m1 = data.train.next_batch(batch_size_m1)
            feed_dict_train_m1 = {x: X_batch_m1, y_true: y_batch_m1}
            
            _, l_batch_m1, pred_m1 = sess.run([optimizer1, loss1, logits1], feed_dict=feed_dict_train_m1)
            los_m1, acc_batch_m1 = sess.run([loss1, accuracy1], feed_dict=feed_dict_train_m1)

            loss_list_m1.append(l_batch_m1)
            acc_list_m1.append(acc_batch_m1)
            
        ## weights of model 1         
        w11 = get_weights_variable(layer_name='hidden11')
        w21 = get_weights_variable(layer_name='hidden21')
        w31 = get_weights_variable(layer_name='hidden31')

        w_out1 = get_weights_variable(layer_name='outputs1')
        
        b11 = get_bias_variable(layer_name='hidden11')
        b21 = get_bias_variable(layer_name='hidden21')
        b31 = get_bias_variable(layer_name='hidden31')
        b_out1 = get_bias_variable(layer_name='outputs1')
        
        weights11 = sess.run(w11)
        weights21 = sess.run(w21)
        weights31 = sess.run(w31)
        weights_out1 = sess.run(w_out1)
        
        bias11 = sess.run(b11)
        bias21 = sess.run(b21)
        bias31 = sess.run(b31)
        bias_out1 = sess.run(b_out1)


        for iteration in range(total_iterations_m2):

            X_batch_m2, y_batch_m2 = data.train.next_batch(batch_size_m2)
            feed_dict_train_m2 = {x: X_batch_m2, y_true: y_batch_m2}
            
            _, l_batch_m2, pred_m2 = sess.run([optimizer2, loss2, logits2], feed_dict=feed_dict_train_m2)
            los_m2, acc_batch_m2 = sess.run([loss2, accuracy2], feed_dict=feed_dict_train_m2)

            loss_list_m2.append(l_batch_m2)
            acc_list_m2.append(acc_batch_m2)
        

        ## weights of model 2         
        w12 = get_weights_variable(layer_name='hidden12')
        w22 = get_weights_variable(layer_name='hidden22')       
        w32 = get_weights_variable(layer_name='hidden32')
        w_out2 = get_weights_variable(layer_name='outputs2')
        
        ## Bias of model 2
        b12 = get_bias_variable(layer_name='hidden12')
        b22 = get_bias_variable(layer_name='hidden22')
        b32 = get_bias_variable(layer_name='hidden32')
        b_out2 = get_bias_variable(layer_name='outputs2')


        weights12 = sess.run(w12)
        weights22 = sess.run(w22)
        weights32 = sess.run(w32)
        weights_out2 = sess.run(w_out2)
        
                
        bias12 = sess.run(b12)
        bias22 = sess.run(b22)
        bias32 = sess.run(b32)
        bias_out2 = sess.run(b_out2)

##


# In[7]:


fig,ax = fig, ax=plt.subplots()

ax.plot(loss_list_m1,label='test_loss')
ax.plot(acc_list_m1,label='test_acc')

plt.ylabel('Loss')
plt.xlabel('Alpha')
ax.legend();


# In[8]:


fig,ax = fig, ax=plt.subplots()

ax.plot(loss_list_m2,label='test_loss')
ax.plot(acc_list_m2,label='test_acc')

plt.ylabel('Loss')
plt.xlabel('Alpha')
ax.legend();


# In[9]:


tf.trainable_variables()


# In[16]:


loss_list_test = []
acc_list_test = []
loss_list_train = []
acc_list_train = []

alpha_arr = np.linspace(-1,2,100)

for i in range(0,alpha_arr.size):

    alpha = alpha_arr[i]
    
    
    #Model 3
    weights13 = tf.Variable((1-alpha)*weights11 + alpha*weights12)
    weights23 = tf.Variable((1-alpha)*weights21 + alpha*weights22)
    weights33 = tf.Variable((1-alpha)*weights31 + alpha*weights32)
    weights_out3 = tf.Variable((1-alpha)*weights_out1 + alpha*weights_out2)
    
    bias13 = tf.Variable((1-alpha)*bias11 + alpha*bias12)
    bias23 = tf.Variable((1-alpha)*bias21 + alpha*bias22)
    bias33 = tf.Variable((1-alpha)*bias31 + alpha*bias32)
    bias_out3 = tf.Variable((1-alpha)*bias_out1 + alpha*bias_out2)
    
    hidden13 = tf.add(tf.matmul(x, weights13),bias13)
    hidden13 = tf.nn.relu(hidden13)

    hidden23 = tf.add(tf.matmul(hidden13, weights23),bias23)
    hidden23 = tf.nn.relu(hidden23)
    
    hidden33 = tf.add(tf.matmul(hidden23, weights33),bias33)
    hidden33 = tf.nn.relu(hidden33)

    logits3 = tf.add(tf.matmul(hidden33, weights_out3),bias_out3)
    logits3 = tf.nn.relu(logits3)
    
    xentropy3 = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits3)
    loss3 = tf.reduce_mean(xentropy3, name="loss3")
    optimizer3 = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss3)

    y_pred3 = tf.nn.softmax(logits=logits3)
    y_pred_cls3 = tf.argmax(y_pred3, dimension=1)

    correct3 = tf.equal(y_pred_cls3, y_true_cls) # correct is a boolean array of size #batch_size#
    accuracy3 = tf.reduce_mean(tf.cast(correct3, tf.float32))
    
    ########
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())   

    
        total_iterations = data.test.num_examples // 10000

        for iteration in range(total_iterations):

            X_test, y_test = data.test.next_batch(10000)
            feed_dict_test = {x: X_test, y_true: y_test}

            l_test = sess.run(loss3, feed_dict=feed_dict_test)
            acc_test = sess.run(accuracy3, feed_dict=feed_dict_test)

            loss_list_test.append(l_test)
            acc_list_test.append(acc_test)    

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())   

    
        total_iterations = 1

        for iteration in range(total_iterations):

            X_train, y_train = data.train.next_batch(10000)
            feed_dict_train = {x: X_train, y_true: y_train}

            l_train = sess.run(loss3, feed_dict=feed_dict_train)
            acc_train = sess.run(accuracy3, feed_dict=feed_dict_train)

            loss_list_train.append(l_train)
            acc_list_train.append(acc_train)    


# In[1]:


import matplotlib 
matplotlib.rcParams['figure.figsize'] = [8, 6]

fig,ax = fig, ax=plt.subplots(2, 1, constrained_layout=True)

fig.suptitle('Visualizing the line between two models', fontsize=16)
               
ax[0].plot(alpha_arr, loss_list_test,label='Testing Loss')
ax[0].plot(alpha_arr, loss_list_train,label='Training Loss')
ax[0].set_ylabel('Loss')
ax[0].set_xlabel('Alpha')
ax[0].legend()

ax[1].plot(alpha_arr, acc_list_test,label='Testing Accuracy')
ax[1].plot(alpha_arr, acc_list_train, label='Training Accuracy')
ax[1].set_ylabel('Accurancy')
ax[1].set_xlabel('Alpha')
ax[1].legend()

plt.show()

