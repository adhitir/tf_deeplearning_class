#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from keras.datasets import cifar10, mnist
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2


# In[2]:


#training_set = x_train

samples = []  # empty array to hold the samples
losses = []  # empty array to hold the losses
Z = 100 #Dimension of the noise vector
beta1=0.5
#beta2=0.999
lr = 0.0002


# In[3]:


# batch_size = 100
# training = True
# noise = tf.placeholder(shape=(None, Z), dtype=tf.float32)
# lrelu_slope= 0.2
# kernel_size=4
# w_init = tf.random_normal_initializer(mean=0.0, stddev=0.02, seed=None)

# input_dense = tf.layers.dense(inputs=noise, units=4*4*256)
# input_volume = tf.reshape(tensor=input_dense, shape=(-1, 4, 4, 256))
# h1 = tf.layers.batch_normalization(inputs=input_volume, training=training)  # batch norm
# h1 = tf.nn.leaky_relu(h1, alpha=lrelu_slope)

# h2 = tf.layers.conv2d_transpose(filters=128, strides=2, kernel_size=4, padding='same', inputs=h1, activation=None, kernel_initializer=w_init)
# h2 = tf.nn.leaky_relu(h2, alpha=lrelu_slope) # leaky relu
        
# h3 = tf.layers.conv2d_transpose(filters=128, strides=2, kernel_size=kernel_size, padding='same', inputs=h2, activation=None, kernel_initializer=w_init)
# h3 = tf.nn.leaky_relu(h3, alpha=lrelu_slope) 

# print(np.shape(h3))

# h4 = tf.layers.conv2d_transpose(filters=128, strides=2, kernel_size=kernel_size, padding='same', inputs=h3, activation=None, kernel_initializer=w_init)
# h4 = tf.nn.leaky_relu(h4, alpha=lrelu_slope) 

# print(np.shape(h4))

# logits = tf.layers.conv2d(filters=3, kernel_size=3, padding='same', inputs=h4, activation=None, kernel_initializer=w_init)

# print(np.shape(logits))


# In[4]:


def generator(input_layer, reuse=False, training=True):

    noise = input_layer #the input into the generator is noise
    lrelu_slope= 0.2
    kernel_size=4
    
    w_init = tf.random_normal_initializer(mean=0.0, stddev=0.02, seed=None)
      
    with tf.variable_scope('generator', reuse=reuse):

        # reshape the noise into a volume
        input_dense = tf.layers.dense(inputs=noise, units=4*4*256)
        input_volume = tf.reshape(tensor=input_dense, shape=(-1, 4, 4, 256))
        h1 = tf.layers.batch_normalization(inputs=input_volume, training=training)  # batch norm
        h1 = tf.nn.leaky_relu(h1, alpha=lrelu_slope)
        # 4x4x256

        # deconv - upsample
        h2 = tf.layers.conv2d_transpose(filters=128, strides=2, kernel_size=kernel_size, padding='same', inputs=h1, activation=None, kernel_initializer=w_init)
        h2 = tf.nn.leaky_relu(h2, alpha=lrelu_slope) # leaky relu
        # 8x8x128

        # deconv - upsample
        h3 = tf.layers.conv2d_transpose(filters=128, strides=2, kernel_size=kernel_size, padding='same', inputs=h2, activation=None, kernel_initializer=w_init)
        h3 = tf.nn.leaky_relu(h3, alpha=lrelu_slope)        
        # 16x16x128

        # deconv - upsample
        h4 = tf.layers.conv2d_transpose(filters=128, strides=2, kernel_size=kernel_size, padding='same', inputs=h3, activation=None, kernel_initializer=w_init)
        h4 = tf.nn.leaky_relu(h4, alpha=lrelu_slope)        
        # 32x32x128

        # deconv - upsample
        logits = tf.layers.conv2d(filters=3, kernel_size=3, padding='same', inputs=h4, activation=None, kernel_initializer=w_init)
        
        out = tf.tanh(x=logits)


        return out
    


# In[5]:


# img_input = tf.placeholder(shape=(None, 32, 32, 3), dtype=tf.float32)


# h1 = tf.layers.conv2d(inputs=img_input, filters=64, strides=2, kernel_size=3, padding='same', kernel_initializer=w_init)
# h1 = tf.nn.leaky_relu(h1, alpha=lrelu_slope)

# print(np.shape(h1))

# #H1 - shape - (batch_size,16,16,64)

# h2 = tf.layers.conv2d(inputs=h1, filters=128, strides=2, kernel_size=3, padding='same', kernel_initializer=w_init)
# h2 = tf.nn.leaky_relu(h2, alpha=lrelu_slope)

# print(np.shape(h2))

# #H2 - shape - (batch_size,8,8,128)

# h3 = tf.layers.conv2d(inputs=h2, filters=128, strides=2, kernel_size=3, padding='same', kernel_initializer=w_init)
# h3 = tf.nn.leaky_relu(h3, alpha=lrelu_slope)

# print(np.shape(h3))

# # #H3 - shape - (batch_size,4,4,256)

# h4 = tf.layers.conv2d(inputs=h3, filters=128, strides=2, kernel_size=3, padding='same', kernel_initializer=w_init)
# h4 = tf.nn.leaky_relu(h4, alpha=lrelu_slope)

# print(np.shape(h4))

# # # #H3 - shape - (batch_size,2,2,512)

# # # # flatten the array
# flatten = tf.reshape(tensor=h3, shape=(-1, 2*2*256))
# dropout = tf.nn.dropout(flatten,rate = 0.4)

# # # # logits
# logits = tf.layers.dense(inputs=flatten, units=1, activation=None, kernel_initializer=w_init)
      
# print(np.shape(logits))


# In[6]:


def discriminator(input_layer, reuse=False):

    img_in = input_layer
    lrelu_slope= 0.2

    w_init = tf.random_normal_initializer(mean=0.0, stddev=0.02, seed=None)

    with tf.variable_scope('discriminator', reuse=reuse):

        h1 = tf.layers.conv2d(inputs=img_in, filters=64, strides=2, kernel_size=3, padding='same', kernel_initializer=w_init)
        h1 = tf.nn.leaky_relu(h1, alpha=lrelu_slope)
        
        #H1 - shape - (batch_size,16,16,64)

        h2 = tf.layers.conv2d(inputs=h1, filters=128, strides=2, kernel_size=3, padding='same', kernel_initializer=w_init)
        h2 = tf.nn.leaky_relu(h2, alpha=lrelu_slope)
        
        #H2 - shape - (batch_size,8,8,128)

        h3 = tf.layers.conv2d(inputs=h2, filters=128, strides=2, kernel_size=3, padding='same', kernel_initializer=w_init)
        h3 = tf.nn.leaky_relu(h3, alpha=lrelu_slope)
        
        #H3 - shape - (batch_size,4,4,256)
        
        h4 = tf.layers.conv2d(inputs=h3, filters=128, strides=2, kernel_size=3, padding='same', kernel_initializer=w_init)
        h4 = tf.nn.leaky_relu(h4, alpha=lrelu_slope)
        
        #H3 - shape - (batch_size,2,2,512)

        # flatten the array
        flatten = tf.reshape(tensor=h3, shape=(-1, 2*2*256))
        dropout = tf.nn.dropout(flatten,0.4)

        # logits
        logits = tf.layers.dense(inputs=flatten, units=1, activation=None, kernel_initializer=w_init)
                
        out = tf.sigmoid(x=logits)

        return out


# In[7]:


def __next_batch(data, batch_size=128):

    # get the number of partitions
    number_of_partitions = data.shape[0]//batch_size

    # shuffle the examples
    np.random.shuffle(data)

    # partition the examples
    for batch in np.array_split(data[:number_of_partitions*batch_size], number_of_partitions):
        yield batch * 2 - 1  # scale to -1 to 1

def view_samples(epoch, samples, nrows, ncols, figsize=(5, 5)):

    # ge the figure and the axes
    fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols, sharey=True, sharex=True)

    # draw the samples
    for ax, img in zip(axes.flatten(), samples):
        ax.axis('off')
        img = ((img - img.min())*255 / (img.max() - img.min())).astype(np.uint8)
        ax.set_adjustable('box')
        im = ax.imshow(img, aspect='equal')
    
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.subplots_adjust(wspace=0, hspace=0)
    return fig, axes


# In[8]:


def binary_cross_entropy(x, z):
    eps = 1e-12
    return (-(x * tf.log(z + eps) + (1. - x) * tf.log(1. - z + eps)))


# In[9]:


tf.reset_default_graph()

noise_input = tf.placeholder(shape=(None, Z), dtype=tf.float32)
img_input = tf.placeholder(shape=(None, 32, 32, 3), dtype=tf.float32)

# Generate fake images
fake_images = generator(noise_input, reuse=False, training=True)

# Two disciminator objects

# for real images
disc_real_output = discriminator(input_layer=img_input, reuse=False)
        
# for fake images
disc_fake_output = discriminator(input_layer=fake_images, reuse=True) # reuse the variables from the real images


# In[10]:


# get the variables for the generator and discriminator
generator_variables = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
discriminator_variables = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]


# In[11]:


# Apply regularization
d_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), discriminator_variables)
g_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), generator_variables)


# In[12]:


def discriminator_loss(real_output, fake_output):

    real_loss = binary_cross_entropy(tf.ones_like(real_output),real_output)
    fake_loss = binary_cross_entropy(tf.zeros_like(fake_output),fake_output)
    total_loss = tf.reduce_mean(0.5 * (real_loss + fake_loss))
    return total_loss

def generator_loss(fake_output):
    
    gen_loss = tf.reduce_mean(binary_cross_entropy(tf.ones_like(fake_output), fake_output))

    return gen_loss

disc_loss = discriminator_loss(disc_real_output,disc_fake_output)
gen_loss = generator_loss(disc_fake_output)


# In[13]:


# setup the optimizers
# comtrol for the global sample mean and variance

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    generator_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1).minimize(gen_loss + g_reg, var_list=generator_variables)
    discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1).minimize(disc_loss + d_reg, var_list=discriminator_variables)
        


# In[ ]:


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = np.concatenate((x_train, x_test), axis = 0)/255

batch_size=128
epochs= 88


with tf.Session() as sess:

    # initialize the variables
    sess.run(tf.global_variables_initializer())
    
    g_loss_all = []
    d_loss_all = []

    # train the network
    for epoch in tqdm(range(epochs)):
        
#         print('ok')


        for step, batch in enumerate(__next_batch(x_train, batch_size)):

            noise = np.random.uniform(low=-1, high=1, size=(batch_size, Z))
            
            _ = sess.run(generator_optimizer, feed_dict={noise_input: noise, img_input: batch})
#             print('ok')


            _ = sess.run(discriminator_optimizer, feed_dict={noise_input: noise, img_input: batch})

            g_loss, d_loss = sess.run([gen_loss, disc_loss], feed_dict={noise_input: noise, img_input: batch})

            # append all the losses on every iteration
            g_loss_all.append(g_loss)
            d_loss_all.append(d_loss)
            
        #g_epoch = np.mode(g_batch)
        #d_epoch = np.mode(d_batch)

        #losses.append((g_epoch, d_epoch))
        print('GEN loss:', g_loss, 'DIS loss:', d_loss)


        # every second epoch
        if epoch % 2 == 0:
            # sample more noise
            sample_noise = np.random.uniform(low=-1, high=1, size=(1000, Z))

            # generate images
            gen_samples = sess.run(generator(noise_input, reuse=True, training=False), feed_dict={noise_input: sample_noise})

            # append the images to the samples
            samples = gen_samples[1:21]

            # view samples from the last epoch
            _ = view_samples(-1, samples, 4, 5, figsize=(5,5))
            
            plt.show()
            
            
                    
i = 0
for img in gen_samples:
    img1 = ((img - img.min())*255 / (img.max() - img.min())).astype(np.uint8)
    type(img1)
    figname = "dcgan_image{:04d}.jpg".format(i)
    cv2.imwrite('/home/adhitir/class_stuff/Homework3/dcgan_final_pics/'+figname,img1) 
    i=i+1


# In[ ]:


#g_loss = [i[0] for i in losses]
#d_loss = [i[1] for i in losses]


# In[ ]:


import matplotlib.pyplot as plt

plt.plot(d_loss_all)
plt.ylabel('discriminator loss')
plt.xlabel('iterations')
plt.show()
plt.savefig('d_loss.png')


# In[ ]:


plt.plot(g_loss_all)
plt.ylabel('generator loss')
plt.xlabel('iterations')
plt.show()
plt.savefig('g_loss.png')


# In[ ]:


import cv2

i = 0
for img in gen_samples:
    img1 = ((img - img.min())*255 / (img.max() - img.min())).astype(np.uint8)
    type(img1)
    figname = "dcgan_image{:04d}.jpg".format(i)
    cv2.imwrite('/home/adhitir/class_stuff/Homework3/dcgan_final_pics/'+figname,img1) 
    i=i+1


# In[ ]:




