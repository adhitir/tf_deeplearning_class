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


# In[4]:


def get_weights_variable(layer_name):
    # Retrieve an existing variable named 'kernel' in the scope with the given layer_name.

    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('kernel',dtype=tf.float32)

    return variable


# In[5]:


# Train the model 


# In[6]:


from sklearn.decomposition import PCA

n_epochs = 1000
batch_size = 200

weight_for8trains=[[]]
trainings = 8

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

weight_trial1 = []
weight_trial2 = []


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
    


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())     

        for epoch in range(n_epochs):

            l_epoch = 0
            acc_epoch = 0

            total_iterations = data.train.num_examples // batch_size

            for iteration in range(total_iterations):

                X_batch, y_batch = data.train.next_batch(batch_size)

                feed_dict_train = {x: X_batch, y_true: y_batch}

                _, l_batch, pred = sess.run([optimizer, loss, logits], feed_dict=feed_dict_train)

                los, acc_batch = sess.run([loss, accuracy], feed_dict=feed_dict_train)

                l_epoch = l_epoch + l_batch/total_iterations
                acc_epoch = acc_epoch + acc_batch/total_iterations


            loss_list.append(l_epoch)
            acc_list.append(acc_epoch)

            weight_epoch1 = []
            weight_epoch2 = []


            if epoch % 3 == 0:

                ## weights         
                w1 = get_weights_variable(layer_name='hidden1')
                w2 = get_weights_variable(layer_name='hidden2')
                w_out = get_weights_variable(layer_name='outputs')

                weights_list1 = [w1,w2,w_out]
                weights_list2 = [w1]

                w_list1 = sess.run(weights_list1)
                w_list2 = sess.run(weights_list2)

                w_1d_1 = []
                w_1d_2 = []

                for w in w_list1:
                    ar1 = w.reshape(1,-1)
                    w_1d_1.append(ar1)
                    
                ar2 = w_list2.reshape(1,-1)
                w_1d_2.append(ar2)

                w_1d_arr1 = np.hstack(w_1d_1)
                w_1d_arr2 = np.hstack(w_1d_2)

                weight_epoch1 = w_1d_arr1.reshape(1,-1)
                weight_epoch2 = w_1d_arr2.reshape(1,-1)

                #weight_trial = np.concatenate((weight_trial,weight_epoch),axis=0) 
                weight_trial1.append(weight_epoch1)
                weight_trial2.append(weight_epoch2)

        #weight_trial_array = np.vstack(weight_trial)



##


# In[ ]:


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('PCA for full model weights', fontsize = 20)

weight_trial_array1 = np.vstack(weight_trial1)
size1 = weight_trial_array1.shape

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(weight_trial_array1)

i = 0
j = size1[0]//trainings

for iters in range(trainings):
    
    ax.scatter(principalComponents[i:j,0], principalComponents[i:j,1], label=weight_trial_array1[i:j].shape, alpha=0.5)
    
    i = j+1
    j = j + size1[0]//trainings

ax.legend()
plt.show()


# In[7]:


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('PCA for layer 1 weights', fontsize = 20)
    
weight_trial_array2 = np.vstack(weight_trial2)
size2 = weight_trial_array2.shape

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(weight_trial_array2)

i = 0
j = size2[0]//trainings

for iters in range(trainings):
    
    ax.scatter(principalComponents[i:j,0], principalComponents[i:j,1], label=weight_trial_array2[i:j].shape, alpha=0.5)
    
    i = j+1
    j = j + size2[0]//trainings

ax.legend()
plt.show()


# In[8]:


fig,ax = fig, ax=plt.subplots()

ax.plot(loss_list,label='Loss')
ax.plot(acc_list,label='Acc')

plt.ylabel('Loss/Acc')
plt.xlabel('Epoch')
ax.legend();

