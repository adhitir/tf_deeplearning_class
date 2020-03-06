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


import random

def random_labels(y_batch,batch_size):
    
    #random_ytrain = []
    
    a = y_batch[0,:]             
    random.shuffle(a)
    random_ytrain = a 
    
    for i in range (1,batch_size):
        
        a = y_batch[i,:]             
        random.shuffle(a)
        random_ytrain = np.vstack((random_ytrain,a))
        #label_mapping = np.array(label_mapping, dtype=np.float64)
        #label_mapping.tolist()
        #random_ytrain.append(label_mapping)        
    return random_ytrain
#         a = np.array([1, 2, 3, 4], dtype=np.float64)
# a.tolist()
# a.reshape(2,2)


# In[10]:



def random_labels(y_batch,batch_size):
    
    random_train = []

    for i in range (0,batch_size):
        
        a = y_batch[i,:]   
        num_a = np.where(a==1)
        num_a = num_a[0]
        #print(num_a)
        if num_a == 0:
            new_batch = np.array([0,0,0,0,0,0,0,0,0,1],dtype = np.float64)
        if num_a == 1:
            new_batch = np.array([0,0,0,0,0,0,0,0,1,0],dtype = np.float64)
        if num_a == 2:
            new_batch = np.array([0,0,0,0,0,0,0,1,0,0],dtype = np.float64)          
        if num_a == 3:
            new_batch = np.array([0,0,0,0,0,0,1,0,0,0],dtype = np.float64)
        if num_a == 4:
            new_batch = np.array([0,0,0,0,0,1,0,0,0,0],dtype = np.float64)            
        if num_a == 5:
            new_batch = np.array([1,0,0,0,0,0,0,0,0,0],dtype = np.float64)
        if num_a == 6:
            new_batch = np.array([0,1,0,0,0,0,0,0,0,0],dtype = np.float64)
        if num_a == 7:
            new_batch = np.array([0,0,1,0,0,0,0,0,0,0],dtype = np.float64)          
        if num_a == 8:
            new_batch = np.array([0,0,0,1,0,0,0,0,0,0],dtype = np.float64)
        if num_a == 9:
            new_batch = np.array([0,0,0,0,1,0,0,0,0,0],dtype = np.float64)       
        #random.shuffle(a)
        
        random_train.append(new_batch)
        
    return random_train


# In[11]:


# X_batch, y_batch = data.train.next_batch(batch_size)
# a = y_batch[1,:]   
# random_labels(y_batch,batch_size)


# In[12]:


# Train the model 


# In[16]:


n_epochs = 4000
batch_size = 200

total_parameters = 0

tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
y_true = tf.placeholder(tf.float32, shape=[None,num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# Model 
hidden1 = tf.layers.dense(inputs=x, units=100, activation=tf.nn.relu,name='hidden1') 
hidden2 = tf.layers.dense(inputs=hidden1, units=78, activation=tf.nn.relu,name='hidden2')
hidden3 = tf.layers.dense(inputs=hidden2, units=78, activation=tf.nn.relu,name='hidden3')

logits = tf.layers.dense(inputs=hidden3, units=num_classes,name='outputs') 

y_pred = tf.nn.softmax(logits=logits)
y_pred_cls = tf.argmax(y_pred, dimension=1)

xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
loss = tf.reduce_mean(xentropy, name="loss")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

correct = tf.equal(y_pred_cls, y_true_cls) # correct is a boolean array of size #batch_size#
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
########

get_no_of_parameters()

loss_list_train=[]
loss_list_test=[]
acc_list_train=[]
acc_list_test=[]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())     

    for epoch in range(n_epochs):

        total_iterations = data.train.num_examples // batch_size
        
        for iteration in range(total_iterations):

            X_batch, y_batch = data.train.next_batch(batch_size)
            
            new_batch = random_labels(y_batch,batch_size)
            #np.random.shuffle(y_batch)

            feed_dict_train = {x: X_batch, y_true: new_batch}

            _, l, pred = sess.run([optimizer, loss, logits], feed_dict=feed_dict_train)

            l_train, acc_train = sess.run([loss, accuracy], feed_dict=feed_dict_train)

        loss_list_train.append(l_train)

        acc_list_train.append(acc_train)


        total_iterations_test = data.test.num_examples // batch_size

        for iteration in range(total_iterations_test):

            X_batch_test, y_batch_test = data.test.next_batch(batch_size)

            feed_dict_test = {x: X_batch_test, y_true: y_batch_test}

            l_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_test)


        loss_list_test.append(l_test)
        acc_list_test.append(acc_test)

##


# In[17]:


# fig,ax = fig, ax=plt.subplots()

# ax.plot(acc_list_train,label='train_acc')
# ax.plot(acc_list_test,label='test_acc')


# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# ax.legend();


# In[18]:


fig2,ax2 = fig2, ax2=plt.subplots()

ax2.plot(loss_list_test,label='test_loss')
ax2.plot(loss_list_train,label='train_loss')

plt.ylabel('Loss')
plt.xlabel('Epoch')
ax2.legend();


# In[ ]:


import random
batch_size = 3

total_iterations = data.train.num_examples // batch_size

X_batch, y_train = data.train.next_batch(batch_size)

print(y_train)


type(y_train)


# In[ ]:


l1 = [0 0 0 1]
l2 = np.ndarray(l1)
print(l2)


# In[ ]:


a = np.array([1, 2, 3, 4], dtype=np.float64)
a.tolist()
a.reshape(2,2)


# In[ ]:


a = random_labels(3)


# In[ ]:


a


# In[ ]:


a = []
label_mapping = np.array([0,0,0,0,0,0,0,0,0,1],dtype = np.float64)
random.shuffle(label_mapping)
#label_mapping = np.array(label_mapping, dtype=np.float64)
label_mapping.tolist()
# a.append(label_mapping)
label_mapping


# In[ ]:


a = []
a.append([1,2,3])
a.append([1,2,3])
a


# In[ ]:


X_batch, y_batch = data.train.next_batch(batch_size)
print(y_batch)


# In[ ]:


#y_batch = random_labels(batch_size)
X_batch, y_batch = data.train.next_batch(batch_size)

#new_batch = random_labels(y_batch,batch_size)

a = y_batch[0,:]             
random.shuffle(a)
random_ytrain = a 

for i in range (1,batch_size):

    a = y_batch[i,:]             
    random.shuffle(a)
    random_ytrain = np.vstack((random_ytrain,a))
    
random_ytrain


# In[ ]:


X_batch, y_batch = data.train.next_batch(batch_size)

a = y_batch[1,:]             
random.shuffle(a)
random_ytrain = a 
a


# In[ ]:





# In[ ]:




