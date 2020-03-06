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


#Create data for two different functions

tf.set_random_seed(1)
np.random.seed(1)

x1 = np.expand_dims(np.arange(0.0, 3.0, 0.01),1)
y1 = np.cos(x1*x1) + x1 + np.random.normal(0, 0.1, size=x1.shape)

plt.plot(x1,y1)
plt.show


# In[3]:


num_features =  x1.shape[1]

tf.trainable_variables()


# In[4]:


def get_weights_variable(layer_name):
    # Retrieve an existing variable named 'kernel' in the scope with the given layer_name.

    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('kernel',dtype=tf.float64)

    return variable


# In[5]:


# Train the model 


# In[6]:


from sklearn.decomposition import PCA

n_epochs = 10
batch_size = 200

weight_for8trains=[[]]
trainings = 1

# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1) 
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 component PCA', fontsize = 20)


tf.reset_default_graph()

x = tf.placeholder(dtype = tf.float64, shape = ([None,num_features]), name='x')
y_true = tf.placeholder(dtype = tf.float64, shape = ([None,num_features]), name='y')

y_true_cls = tf.argmax(y_true, dimension=1)

hidden1 = tf.layers.dense(inputs=x, units=10, activation=tf.nn.relu,name='hidden1') 
hidden2 = tf.layers.dense(inputs=hidden1, units=31, activation=tf.nn.relu,name='hidden2')
logits = tf.layers.dense(inputs=hidden2, units=num_features,name='outputs') 

y_pred = tf.nn.softmax(logits=logits)
y_pred_cls = tf.argmax(y_pred, dimension=1)

loss = tf.losses.mean_squared_error(y_true, logits)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

print(hidden1)
print(hidden2)
print(logits)

correct = tf.equal(y_pred_cls, y_true_cls) # correct is a boolean array of size #batch_size#
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

loss_list=[]
acc_list=[]

w_out = get_weights_variable(layer_name='outputs')

grads = tf.gradients(loss, w_out)[0]
hessian = tf.reduce_sum(tf.hessians(loss, w_out)[0], axis = 2)

grad_norm=[]

grad_all_old = 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())   
    
    for epoch in range(1, n_epochs):

        grad = 0

        for iteration in range(20000):

            feed_dict_train = {x: x1, y_true: y1}

            _, l, pred = sess.run([optimizer, loss, logits], feed_dict=feed_dict_train)
            los, acc = sess.run([loss, accuracy], feed_dict=feed_dict_train)

            grads_vals, hess_vals = sess.run([grads, hessian], feed_dict=feed_dict_train)

            #l_epoch = l_epoch + l_batch/total_iterations
            #acc_epoch = acc_epoch + acc_batch/total_iterations

            loss_list.append(l)
            acc_list.append(acc)

            grads_vals_reshape = grads_vals.reshape(-1,)

            grad_all = 0
            grad = 0

            grad_all = np.linalg.norm(grads_vals_reshape)

            grad_norm.append(grad_all)


#             if grad_all < grad_all_old:
#                 w1 = get_weights_variable(layer_name='hidden1')
#                 w2 = get_weights_variable(layer_name='hidden2')
#                 w_out = get_weights_variable(layer_name='outputs')

#                 weights_list = [w1,w2,w_out]

#                 w_list = sess.run(weights_list)


#                 w1 = []

#                 for w in w_list:
#                     ar1 = w.reshape(1,-1)
#                     w1.append(ar1)

#                 w_arr1 = np.hstack(w1)

#                 min_weight = w_arr1.reshape(1,-1)

#             grad_all_old = grad_all


                
                
            #loss_list.append(l_epoch)
            #acc_list.append(acc_epoch)


# In[7]:


import matplotlib 
matplotlib.rcParams['figure.figsize'] = [8, 6]

fig,ax = fig, ax=plt.subplots(2, 1, constrained_layout=True)

fig.suptitle('Loss and Gradient Norm for f(x)', fontsize=16)
               
ax[0].plot(loss_list,label='Loss')
ax[0].set_ylabel('Loss')
ax[0].set_xlabel('Iteration')
ax[0].legend();

ax[1].plot(grad_norm,label='Gradient 2-Norm')
ax[1].set_yscale('log')
ax[1].set_ylabel('Gradient 2-Norm')
ax[1].set_xlabel('Iteration')
ax[1].legend();

plt.show()


# In[8]:


# ## weights of model when grad_norm is zero

# w1 = get_weights_variable(layer_name='hidden11')
# w2 = get_weights_variable(layer_name='hidden21')
# w_out = get_weights_variable(layer_name='outputs1')

# weights_list = [w1,w2,w_out]
# w_list = sess.run(weights_list)


# w1 = []

# for w in w_list:
#     ar1 = w.reshape(1,-1)
#     w1.append(ar1)

# w_arr1 = np.hstack(w1)

# weight_epoch1 = w_arr1.reshape(1,-1)


# In[ ]:


fig,ax = fig, ax=plt.subplots()

ax.plot(loss_list,label='Loss')
ax.plot(acc_list,label='Acc')

plt.ylabel('Loss/Acc')
plt.xlabel('Epoch')
ax.legend();

