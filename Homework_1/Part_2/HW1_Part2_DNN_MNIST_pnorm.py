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


# Train the model 


# In[6]:


n_epochs = 10
batch_size = 200

weight_for8trains=[[]]
trainings = 1


weight_trial = []

for i in range(trainings):
    
    tf.reset_default_graph()
    
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
    y_true = tf.placeholder(tf.float32, shape=[None,num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)
    
    hidden1 = tf.layers.dense(inputs=x, units=300, activation=tf.nn.relu,name='hidden1') 
    hidden2 = tf.layers.dense(inputs=hidden1, units=100, activation=tf.nn.relu,name='hidden2')
    logits = tf.layers.dense(inputs=hidden2, units=num_classes,name='outputs') 

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

    loss_list=[]
    acc_list=[]
    
    w_out = get_weights_variable(layer_name='outputs')

    grads = tf.gradients(loss, w_out)[0]
    hessian = tf.reduce_sum(tf.hessians(loss, w_out)[0], axis = 2)
    
    grad_norm=[]
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())     

        for epoch in range(n_epochs):

            total_iterations = data.train.num_examples // batch_size
            
            grad = 0

            for iteration in range(total_iterations):

                X_batch, y_batch = data.train.next_batch(batch_size)

                feed_dict_train = {x: X_batch, y_true: y_batch}

                _, l_batch, pred = sess.run([optimizer, loss, logits], feed_dict=feed_dict_train)

                los, acc_batch = sess.run([loss, accuracy], feed_dict=feed_dict_train)

                ## weights         
                #w1 = get_weights_variable(layer_name='hidden1')
                #w2 = get_weights_variable(layer_name='hidden2')               

                
                grads_vals, hess_vals = sess.run([grads, hessian], feed_dict=feed_dict_train)
                
                #l_epoch = l_epoch + l_batch/total_iterations
                #acc_epoch = acc_epoch + acc_batch/total_iterations
                
                loss_list.append(l_batch)
                acc_list.append(acc_batch)
                
                grads_vals_reshape = grads_vals.reshape(-1,)
                
                grad_all = 0
                grad = 0
                
                grad_all = np.linalg.norm(grads_vals_reshape)

#                 for p in range (0,grads_vals_reshape.shape[0]):
#                     grad = grad + grads_vals_reshape[p]*grads_vals_reshape[p]
#                     #grad_all = grad_all + grad
#                     #print(grad)

#                 grad_all = np.sqrt(grad)
                                
                grad_norm.append(grad_all)
                
                #print(iteration)
                
            #print(grad)

            #loss_list.append(l_epoch)
            #acc_list.append(acc_epoch)


# In[9]:


import matplotlib 
matplotlib.rcParams['figure.figsize'] = [8, 6]

fig,ax = fig, ax=plt.subplots(2, 1, constrained_layout=True)

fig.suptitle('Loss and Gradient Norm for MINST on a DNN', fontsize=16)
               
ax[0].plot(loss_list,label='Loss')
ax[0].set_title('Loss')
ax[0].set_ylabel('Loss')
ax[0].set_xlabel('Iteration')
ax[0].legend();

ax[1].plot(grad_norm,label='Gradient 2-Norm')
ax[1].set_yscale('log')
ax[1].set_ylabel('Gradient 2-Norm')
ax[1].set_xlabel('Iteration')
ax[1].legend();

plt.show()


# In[ ]:


fig,ax = fig, ax=plt.subplots()

ax.plot(loss_list,label='Loss')
ax.plot(acc_list,label='Acc')

plt.ylabel('Loss/Acc')
plt.xlabel('Epoch')
ax.legend();

