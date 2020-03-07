#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
tf.logging.set_verbosity(tf.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}
tf.reset_default_graph()


# In[2]:


#Create data for two different functions

tf.set_random_seed(1)
np.random.seed(1)

x1 = np.expand_dims(np.arange(0.0, 3.0, 0.01),1)
y1 = np.cos(x1*x1) + x1 + np.random.normal(0, 0.1, size=x1.shape)

plt.plot(x1,y1)
plt.show
x1.shape


# In[3]:


num_features =  x1.shape[1]


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


def get_weights_variable(layer_name):
    # Retrieve an existing variable named 'kernel' in the scope with the given layer_name.

    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('kernel',dtype=tf.float64)

    return variable


# In[6]:


# # Train the model 

# n_epochs = 5
# batch_size = 500


# x = tf.placeholder(dtype = tf.float64, shape = ([None,num_features]), name='x')
# y_true = tf.placeholder(dtype = tf.float64, shape = ([None,num_features]), name='y')

# y_true_cls = tf.argmax(y_true, dimension=1)

# hidden1 = tf.layers.dense(inputs=x, units=10, activation=tf.nn.relu,name='hidden1') 
# hidden2 = tf.layers.dense(inputs=hidden1, units=31, activation=tf.nn.relu,name='hidden2')
# logits = tf.layers.dense(inputs=hidden2, units=num_features,name='outputs') 

# y_pred = tf.nn.softmax(logits=logits)
# y_pred_cls = tf.argmax(y_pred, dimension=1)

# print(hidden1)
# print(hidden2)
# print(logits)

# loss_list=[]

# min_ratio_list=[]


# In[7]:


loss_list=[]
min_ratio_list=[]

n_epochs = 10
batch_size = 200


x = tf.placeholder(dtype = tf.float64, shape = ([None,num_features]), name='x')
y_true = tf.placeholder(dtype = tf.float64, shape = ([None,num_features]), name='y')

y_true_cls = tf.argmax(y_true, dimension=1)

hidden1 = tf.layers.dense(inputs=x, units=10, activation=tf.nn.relu,name='hidden1') 
hidden2 = tf.layers.dense(inputs=hidden1, units=31, activation=tf.nn.relu,name='hidden2')
logits = tf.layers.dense(inputs=hidden2, units=num_features,name='outputs') 

y_pred = tf.nn.softmax(logits=logits)
#y_pred_cls = tf.argmax(y_pred, dimension=1)

loss = tf.losses.mean_squared_error(y_true, logits)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
w_out = get_weights_variable(layer_name='outputs')
grads = tf.gradients(loss, w_out)[0]
hessian = tf.reduce_sum(tf.hessians(loss, w_out)[0], axis = 2)

loss_grad = tf.norm(grads,ord=2)
#grads = tf.gradients(loss, w_out)[0]
hessian_grad = tf.reduce_sum(tf.hessians(loss_grad, w_out)[0], axis = 2)

print(hidden1)
print(hidden2)
print(logits)

loss_trend = []

grad_norm=[]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #tf.reset_default_graph()    
    
    for i in range(10):
        grad_all_old = 1

        for epoch in range(500):

            feed_dict_train = {x: x1, y_true: y1}

            _, l, pred = sess.run([optimizer, loss, logits], feed_dict=feed_dict_train)

            loss_trend.append(l)

            grads_vals, hess_vals = sess.run([grads, hessian], feed_dict=feed_dict_train)

            grads_vals_reshape = grads_vals.reshape(-1,)

            grad_all = 0

            grad_all = np.linalg.norm(grads_vals_reshape)

            grad_norm.append(grad_all)
            
            if grad_all < grad_all_old:
                hess_min = hess_vals
                grad_all_old = grad_all
                    
                    
        print(grad_all_old)
        w,v= np.linalg.eig(hess_vals)
        no_of_eig_positive = len([i for i in w if i > 0]) 
        no_of_eig = len(w)
        min_ratio = no_of_eig_positive/no_of_eig

        loss_list.append(l)
        min_ratio_list.append(min_ratio)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss_grad)

    for i in range(90):
        
        grad_all_old = 1

        for epoch in range(500):

            feed_dict_train = {x: x1, y_true: y1}

            _, l, pred = sess.run([optimizer, loss_grad, logits], feed_dict=feed_dict_train)

            loss_trend.append(l)

            grads_vals, hess_vals = sess.run([grads,hessian_grad], feed_dict=feed_dict_train)

            grads_vals_reshape = grads_vals.reshape(-1,)

            grad_all = 0

            grad_all = np.linalg.norm(grads_vals_reshape)

            #grad_all = sess.run(loss2, feed_dict=feed_dict_train)

            grad_norm.append(grad_all)
            
            if grad_all < grad_all_old:
                hess_min = hess_vals
                grad_all_old = grad_all
                    
                    
        print(grad_all_old)
        w,v= np.linalg.eig(hess_vals)
        no_of_eig_positive = len([i for i in w if i > 0]) 
        no_of_eig = len(w)
        min_ratio = no_of_eig_positive/no_of_eig

        loss_list.append(l)
        min_ratio_list.append(min_ratio)
    
    
            
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


# In[8]:


hess_vals


# In[9]:


# w,v= np.linalg.eig(hess_vals)
# no_of_eig_positive = len([i for i in w if i > 0]) 
# no_of_eig = len(w)
# print(no_of_eig_positive)
# print(no_of_eig)
# min_ratio = no_of_eig_positive/no_of_eig
# print(min_ratio)


# In[10]:


import matplotlib 
matplotlib.rcParams['figure.figsize'] = [8, 6]

fig,ax = fig, ax=plt.subplots()

fig.suptitle('Loss and Gradient Norm for f(x)', fontsize=16)
               
ax.scatter(min_ratio_list, loss_list)
ax.set_title('Loss')
ax.set_ylabel('Loss')
ax.set_xlabel('min_ratio')
ax.legend();


# In[11]:


import matplotlib 
matplotlib.rcParams['figure.figsize'] = [8, 6]

fig,ax = fig, ax=plt.subplots(2, 1, constrained_layout=True)

fig.suptitle('Loss and Gradient Norm for f(x)', fontsize=16)
               
ax[0].plot(loss_trend,label='Loss')
ax[0].set_ylabel('Loss')
ax[0].set_xlabel('Iteration')
ax[0].legend();

ax[1].plot(grad_norm,label='Gradient normal (2-Norm) of the loss')
ax[1].set_yscale('log')
ax[1].set_ylabel('Gradient 2-Norm')
ax[1].set_xlabel('Iteration')
ax[1].legend();


# In[12]:


np.shape(loss_trend)


# In[13]:


# ## weights of model when grad_norm is zero
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())


#     w1 = get_weights_variable(layer_name='hidden1')
#     w2 = get_weights_variable(layer_name='hidden2')
#     w_out = get_weights_variable(layer_name='outputs')

#     weights_list = [w1,w2,w_out]
#     w_list = sess.run(weights_list)


#     w1 = []

#     for w in w_list:
#         ar1 = w.reshape(1,-1)
#         w1.append(ar1)

#     w_arr1 = np.hstack(w1)

#     weight_epoch1 = w_arr1.reshape(1,-1)

