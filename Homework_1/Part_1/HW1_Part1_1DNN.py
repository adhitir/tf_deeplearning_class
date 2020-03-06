#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()
tf.reset_default_graph()


# In[2]:


import numpy as np

#Create data for two different functions
tf.set_random_seed(1)
np.random.seed(1)

x1 =np.expand_dims(np.arange(0.0, 3.0, 0.01),1)
#y1 = np.cos(x1*x1) + x1 + np.random.normal(0, 0.1, size=x1.shape)
y1 = np.sin(2*x1) + x1 + np.random.normal(0, 0.1, size=x1.shape)
plt.plot(x1,y1)
plt.show


# In[3]:


# Creating placeholders
num_features =  x1.shape[1]

x_input = tf.placeholder(dtype = tf.float64, shape = ([None,num_features]), name='x')
y_output = tf.placeholder(dtype = tf.float64, shape = ([None,num_features]), name='y')


# In[4]:


#Shallow network
h11 = tf.layers.dense(inputs=x_input, units=10, activation=tf.nn.relu) 
h21 = tf.layers.dense(inputs=h11, units=31, activation=tf.nn.relu)
output1 = tf.layers.dense(inputs=h21, units=num_features)                       # output layer


# In[5]:


#Deep network
h12 = tf.layers.dense(inputs=x_input, units=10, activation=tf.nn.relu)
h22 = tf.layers.dense(inputs=h12, units=5, activation=tf.nn.relu)
h32 = tf.layers.dense(inputs=h22, units=5, activation=tf.nn.relu)
h42 = tf.layers.dense(inputs=h32, units=5, activation=tf.nn.relu)
h52 = tf.layers.dense(inputs=h42, units=5, activation=tf.nn.relu)
h62 = tf.layers.dense(inputs=h52, units=5, activation=tf.nn.relu)
h72 = tf.layers.dense(inputs=h62, units=5, activation=tf.nn.relu)
h82 = tf.layers.dense(inputs=h72, units=5, activation=tf.nn.relu)
h92 = tf.layers.dense(inputs=h82, units=20, activation=tf.nn.relu)

output2 = tf.layers.dense(inputs=h92, units=num_features) 


# In[6]:


# Medium network
h13 = tf.layers.dense(inputs=x_input, units=10, activation=tf.nn.relu)
h23 = tf.layers.dense(inputs=h13, units=5,activation=tf.nn.relu)
h33 = tf.layers.dense(inputs=h23, units=5,activation=tf.nn.relu)
h43 = tf.layers.dense(inputs=h33, units=5, activation=tf.nn.relu)
h53 = tf.layers.dense(inputs=h43, units=37, activation=tf.nn.relu)
output3 = tf.layers.dense(inputs=h53, units=num_features) 


# In[7]:


loss1 = tf.losses.mean_squared_error(y_output, output1)   # compute cost
loss2 = tf.losses.mean_squared_error(y_output, output2)   # compute cost
loss3 = tf.losses.mean_squared_error(y_output, output3)   # compute cost


# In[8]:


optimizer1 = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss1)
optimizer2 = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss2)
optimizer3 = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss3)


# In[9]:


# total_parameters = 0
# for variable in tf.trainable_variables():
# # shape is an array of tf.Dimension
#     shape = variable.get_shape()
#     variable_parameters = 1
#     for dim in shape:
#         variable_parameters *= dim.value
#     total_parameters += variable_parameters
# print(total_parameters)


# In[10]:


#sess = tf.Session()                                 # control training and others


# In[11]:


#sess.run(tf.global_variables_initializer())         # initialize var in graph


# In[12]:


loss1_list=[]
loss2_list=[]
loss3_list=[]


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for iteration in range(20000):
        # train and net output
        _, l1, pred1 = sess.run([optimizer1, loss1, output1], feed_dict={x_input: x1, y_output: y1})
        _, l2, pred2 = sess.run([optimizer2, loss2, output2], feed_dict={x_input: x1, y_output: y1})
        _, l3, pred3 = sess.run([optimizer3, loss3, output3], feed_dict={x_input: x1, y_output: y1})

        loss1_list.append(l1)
        loss2_list.append(l2)
        loss3_list.append(l3)

        #if iteration % 20 == 0:

            # plot and show learning process
            #plt.cla()
            #plt.scatter(x1, y1)
            #plt.plot(x1, pred, 'r-', lw=5)
            #plt.text(0.5, 0, 'Loss=%.4f' % l, fontdict={'size': 20, 'color': 'red'})
            #plt.pause(0.1)
            
    YP1 = sess.run(output1,feed_dict={x_input:x1})
    YP2 = sess.run(output2,feed_dict={x_input:x1})
    YP3 = sess.run(output3,feed_dict={x_input:x1})


# In[13]:


fig, ax = fig, ax = plt.subplots()

ax.plot(x1,y1,label='f(x)')
ax.plot(x1,YP1,label='shallow')
ax.plot(x1,YP2,label='deep')
ax.plot(x1,YP3,label='medium')

plt.ylabel('y_pred')
plt.xlabel('x')

leg = ax.legend();


# In[14]:


fig2, ax2 = fig2, ax2 = plt.subplots()

ax2.plot(loss1_list,label='shallow')
ax2.plot(loss2_list, label='deep')
ax2.plot(loss3_list,label='medium')

plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.ylim((0,0.4))

leg2 = ax2.legend();




