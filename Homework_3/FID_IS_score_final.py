#!/usr/bin/env python
# coding: utf-8

# In[1]:



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.p

from keras.datasets import cifar10
import numpy as np
from six.moves import urllib
import tensorflow as tf
import glob
#import scipy.misc
import math
import sys
import functools
import time
#from tensorflow.python.ops import array_ops
#from tensorflow.python.ops import functional_ops
from glob import glob
import cv2
import os.path
import tarfile
global softmax
from scipy import linalg


# In[ ]:





# In[2]:


def get_inception_score(images, splits=10):
    assert(type(images) == list)
    assert(type(images[0]) == np.ndarray)
    assert(len(images[0].shape) == 3)
    assert(np.max(images[0]) > 10)
    assert(np.min(images[0]) >= 0.0)
    inps = []
    for img in images:
        img = img.astype(np.float32)
        inps.append(np.expand_dims(img, 0))
    bs = 1
    with tf.Session() as sess:
        softmax = _get_inception_logits(sess)
        preds = []
        n_batches = int(math.ceil(float(len(inps)) / float(bs)))
        for i in range(n_batches):
            sys.stdout.write(".")
            sys.stdout.flush()
            inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
            inp = np.concatenate(inp, 0)
            pred = sess.run(softmax, {'ExpandDims:0': inp})

            preds.append(pred)
        preds = np.concatenate(preds, 0)
        scores = []
        for i in range(splits):
            part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        return np.mean(scores), np.std(scores)


# In[3]:


def _get_activation_stats(sess, images, verbose=False):
    
    # Get activations of the pool32 layer
    pool3_layer = _get_inception_layer(sess)
    
    inps = []
    for img in images:
        img = img.astype(np.float32)
        inps.append(np.expand_dims(img, 0))
        
    bs = 1
    n_imgs = math.ceil(len(inps))
    n_batches = int(math.ceil(float(len(inps)) / float(bs)))
    
    #pred_arr = np.empty((len(inps)*2048))
    pred_arr = []
    
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
        inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
        inp = np.concatenate(inp, 0)
        pred = sess.run(pool3_layer, {'ExpandDims:0': inp})
        
        pred = pred.reshape(bs,-1)
        pred_arr.append(pred)
        
    pred_arr = np.concatenate(pred_arr, 0)

    print(np.shape(pred_arr))
    
    assert(np.shape(np.empty((n_imgs,2048))) == np.shape(pred_arr))
    if verbose:
        print(" done")

    mu = np.mean(pred_arr, axis = 0)
    sigma = np.cov(pred_arr, rowvar=False)
        
    return mu, sigma
            
def get_FID_score(gen_images, real_images, eps = 1e-6):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    
        mu1, s1 = _get_activation_stats(sess,gen_images)
        mu2, s2 = _get_activation_stats(sess,real_images)
        
    
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(s1)
    sigma2 = np.atleast_2d(s2)
    
    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    


# In[4]:


def get_images(filename):
    x = cv2.imread(filename)
    return x


# In[5]:


def _create_inception_graph(MODEL_DIR):
    with tf.gfile.FastGFile(os.path.join(MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    
    

def _get_inception_layer(sess):
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')
    #print("The last layer of the inception model has this shape:" % np.shape(pool3))
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            shape = [s.value for s in shape]
            new_shape = []
            for j, s in enumerate(shape):
                if s == 1 and j == 0:
                    new_shape.append(None)
                else:
                    new_shape.append(s)
            o.set_shape(tf.TensorShape(new_shape))
    return pool3
    
    
def _get_inception_logits(sess):
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')
    print("The last layer of the inception model has this shape:" % np.shape(pool3))
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            shape = [s.value for s in shape]
            new_shape = []
            for j, s in enumerate(shape):
                if s == 1 and j == 0:
                    new_shape.append(None)
                else:
                    new_shape.append(s)
            o.set_shape(tf.TensorShape(new_shape))
    w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
    logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
    softmax = tf.nn.softmax(logits)
        
    return softmax


# In[6]:


# Path to Inception model

MODEL_DIR = '/tmp/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None

# Path to generated images:

#filenames = glob(os.path.join('/home/adhitir/class_stuff/Homework3/wgan_final_pics/', '*.*'))
filenames = glob(os.path.join('/home/adhitir/class_stuff/Homework3/dcgan_final_pics/', '*.*'))


#Download Inception model:

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
filename = DATA_URL.split('/')[-1]
filepath = os.path.join(MODEL_DIR, filename)
if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)


# In[7]:


# 100 images generated by the DCGAN or WGAN
gen_images = [get_images(filename) for filename in filenames]

gen_images = gen_images[:10]

#100 randomly selected images from the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = np.concatenate((x_train, x_test),axis = 0)
rand = np.random.permutation(len(x_train))
rand_x = x_train[rand]

real_images = rand_x[:10]

_create_inception_graph(MODEL_DIR)

IS = get_inception_score(gen_images, splits=10)

#FID = get_FID_score(gen_images, real_images)

FID = get_FID_score(real_images, gen_images)


# In[8]:


# inps = []
# for img in gen_images:
#     img = img.astype(np.float32)
#     inps.append(np.expand_dims(img, 0))
    
# print()


# In[9]:


# print(np.shape(inps))
# bs = 1
# i = 1
# inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
# inp = np.concatenate(inp, 0)

# np.shape(inp)

# n_imgs = int(math.ceil((len(inps))))

# print(type(n_imgs))
# np.shape(np.empty((100,2048)))


# In[10]:


print("FID:", FID)
print("IS : ", IS)


# In[11]:


#images[0:1]


# In[ ]:




