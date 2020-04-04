from __future__ import absolute_import, division, print_function, unicode_literals
import random
import json
import os
import time
import tensorflow as tf
from tensorflow.contrib import rnn
from tqdm import tqdm


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#from sklearn.model_selection import train_test_split
import tensorflow.contrib.legacy_seq2seq as seq2seq
from utilities import show_graph
#from util import inv_sigmoid, linear_decay, dec_print_train, dec_print_val, dec_print_test

import unicodedata
import re
import numpy as np
import os
import io
import time
import collections
import json
import sys
import string

import argparse

random.seed(0)
np.random.seed(0)
tf.set_random_seed(0)

n_inputs        = 4096
n_hidden        = 600
val_batch_size  = 100 #100
n_frames        = 80
max_caption_len = 50
forget_bias_red = 1.0
forget_bias_gre = 1.0
dropout_prob    = 0.5
n_attention     = n_hidden
num_epochs = 10
num_display_steps = 15
num_saver_epochs = 3


filename_train = 'MLDS_hw2_1_data/training_label.json'

ckpt_path = 'saved_model/trained_model.ckpt-959'


saver_path = 'saved_model/'

batch_size_test = 100

batch_size = 250

MAX_WORDS = max_caption_len #max number of words in a caption
n_features = n_inputs
no_of_frames = n_frames
sizeof_sentence= MAX_WORDS
learning_rate = 0.0001
n_hidden = n_hidden

special_tokens  = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
phases = {'train': 0, 'val': 1, 'test': 2}

#The following function was taken from: https://github.com/AdrianHsu/S2VT-seq2seq-video-captioning-attention

class S2VT:
    def __init__(self, vocab_num = 0,lr = 1e-4):

        self.vocab_num = vocab_num
        self.learning_rate = lr

     
    def build_model(self, feat, captions=None, cap_len=None, sampling=None, phase=0):

        weights = {
            'W_feat': tf.Variable( tf.random_uniform([n_inputs, n_hidden], -0.1, 0.1), name='W_feat'), 
            'W_dec': tf.Variable(tf.random_uniform([n_hidden, self.vocab_num], -0.1, 0.1), name='W_dec')
        }
        biases = {
            'b_feat':  tf.Variable( tf.zeros([n_hidden]), name='b_feat'),
            'b_dec': tf.Variable(tf.zeros([self.vocab_num]), name='b_dec')
        }   
        embeddings = {
         'emb': tf.Variable(tf.random_uniform([self.vocab_num, n_hidden], -0.1, 0.1), name='emb')
        }

        batch_size = tf.shape(feat)[0]

        # cap_len: (250, 1) -> (250, 50)
        cap_mask = tf.sequence_mask(cap_len, max_caption_len, dtype=tf.float32)
     
        if phase == phases['train']: #  add noise
            noise = tf.random_uniform(tf.shape(feat), -0.1, 0.1, dtype=tf.float32)
            feat = feat + noise

        if phase == phases['train']:
            feat = tf.nn.dropout(feat, dropout_prob)

        feat = tf.reshape(feat, [-1, n_inputs])
        image_emb = tf.matmul(feat, weights['W_feat']) + biases['b_feat']
        image_emb = tf.reshape(image_emb, [-1, n_frames, n_hidden])
        image_emb = tf.transpose(image_emb, perm=[1, 0, 2])
        
        with tf.variable_scope('LSTM1'):
            lstm_red = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=forget_bias_red, state_is_tuple=True)
            if phase == phases['train']:
                lstm_red = tf.contrib.rnn.DropoutWrapper(lstm_red, output_keep_prob=dropout_prob)    
        with tf.variable_scope('LSTM2'):
            lstm_gre = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=forget_bias_gre, state_is_tuple=True)
            if phase == phases['train']:
                lstm_gre = tf.contrib.rnn.DropoutWrapper(lstm_gre, output_keep_prob=dropout_prob)    

        state_red = lstm_red.zero_state(batch_size, dtype=tf.float32)
        state_gre = lstm_gre.zero_state(batch_size, dtype=tf.float32)

        padding = tf.zeros([batch_size, n_hidden])

#         h_src = []
        for i in range(0, n_frames):
            with tf.variable_scope("LSTM1"):
                output_red, state_red = lstm_red(image_emb[i,:,:], state_red)
            
            with tf.variable_scope("LSTM2"):
                output_gre, state_gre = lstm_gre(tf.concat([padding, output_red], axis=1), state_gre)
#                 h_src.append(output_gre) # even though padding is augmented, output_gre/state_gre's shape not change

#         h_src = tf.stack(h_src, axis = 0)

        bos = tf.ones([batch_size, n_hidden])
        padding_in = tf.zeros([batch_size, n_hidden])

        logits = []
        max_prob_index = None

        

        cross_ent_list = []
        for i in range(0, max_caption_len):

            with tf.variable_scope("LSTM1"):
                output_red, state_red = lstm_red(padding_in, state_red)

            if i == 0:
                with tf.variable_scope("LSTM2"):
                    con = tf.concat([bos, output_red], axis=1)
                    output_gre, state_gre = lstm_gre(con, state_gre)
            else:
                if phase == phases['train']:
                    if sampling[i] == True:
                        feed_in = captions[:, i - 1]
                    else:
                        feed_in = tf.argmax(logit_words, 1)
                else:
                    feed_in = tf.argmax(logit_words, 1)
                with tf.device("/cpu:0"):
                    embed_result = tf.nn.embedding_lookup(embeddings['emb'], feed_in)
                with tf.variable_scope("LSTM2"):
                    con = tf.concat([embed_result, output_red], axis=1)
                    output_gre, state_gre = lstm_gre(con, state_gre)

            logit_words = tf.matmul(output_gre, weights['W_dec']) + biases['b_dec']
            logits.append(logit_words)

#             if phase != phases['test']:
            labels = captions[:, i]
            one_hot_labels = tf.one_hot(labels, self.vocab_num, on_value = 1, off_value = None, axis = 1) 
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=one_hot_labels)
            cross_entropy = cross_entropy * cap_mask[:, i]
            cross_ent_list.append(cross_entropy)
        
        loss = 0.0
#         if phase != phases['test']:
        cross_entropy_tensor = tf.stack(cross_ent_list, 1)
        loss = tf.reduce_sum(cross_entropy_tensor, axis=1)
        loss = tf.divide(loss, tf.cast(cap_len, tf.float32))
        loss = tf.reduce_mean(loss, axis=0)

        logits = tf.stack(logits, axis = 0)
        logits = tf.reshape(logits, (max_caption_len, batch_size, self.vocab_num))
        logits = tf.transpose(logits, [1, 0, 2])
        
        summary = None
        if phase == phases['train']:
            summary = tf.summary.scalar('training_loss', loss)
        elif phase == phases['val']:
            summary = tf.summary.scalar('validation_loss', loss)
            

        return logits, loss, summary

    def inference(self, logits):
        
        #print('using greedy search...')
        dec_pred = tf.argmax(logits, 2)
        return dec_pred

    def optimize(self, loss_op):

        params = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(self.learning_rate)#.minimize(loss_op)
        gradients, variables = zip(*optimizer.compute_gradients(loss_op))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        train_op = optimizer.apply_gradients(zip(gradients, params))

        return train_op
    
    
def tokenize(line,token='word'):
    if token == 'word':
        return [line.split(' ')]
    elif token == 'char':
        return [list(line)]
    else:
        print('ERROR: unknown token type '+token)
        
def count_tokens(tokanized_sentences):
    # Flatten a list of token lists into a list of tokens
    tokens = [tk for line in tokanized_sentences for tk in line]
    return collections.Counter(tokens)      
        
def parse_data_into_lists(filename,batch_size,feat_filepath, index2token, tokens):
    
    with open(filename, 'r') as f:
        datastore = json.load(f)
           
    #sentence_set = extract_sentences(filename)
    
    mult_vids = []
    all_sents = []
    all_enc_sents = []
    all_cap_len = []
    all_ids = []
    
    for data in datastore:
        
        sentences = data["caption"]
        sentences = [word.lower() for word in sentences] #Normalize the case
        table = str.maketrans('', '', string.punctuation) #Normalize the punctuation
        sentences = [word.translate(table) for word in sentences]
        
        num_sent = len(sentences)
        
        all_sents.extend(sentences)
        
        enc_sents = []
        
        for sentence in sentences:
            
            #print(sentence)
            tokenized_sentence, encoded_sentence, cap_len = num_encode(sentence,index2token,tokens)
            #print(tokenized_sentence)
            #print(encoded_sentence)
            #print(cap_len)
            #print(encoded_sentence)
            encoded_sentence = list(encoded_sentence)

            enc_sents.append(encoded_sentence)
            all_cap_len.append(cap_len)
            
        all_enc_sents.extend(enc_sents)

        
        video_id = data["id"]
        features = np.load(feat_filepath.format(video_id))
        
        for n in range(0,num_sent):
            mult_vids.append(features)
            all_ids.append(video_id)
        
        print("id: " + str(video_id) + " processed")

            
    return mult_vids, all_sents, all_enc_sents, all_cap_len, all_ids


def parse_data_into_batches(filename,batch_size,feat_filepath, index2token, tokens, mult_vids, all_sents, all_enc_sents, all_cap_len, all_ids):

    with open(filename, 'r') as f:
        datastore = json.load(f)
           
    vid_batch = {}
    sent_batch = {}
    enc_sent_batch = {}
    cap_len_batch = {}
    id_batch = {}

    random.Random(30).shuffle(mult_vids)
    random.Random(30).shuffle(all_sents)
    random.Random(30).shuffle(all_enc_sents)
    random.Random(30).shuffle(all_cap_len)
    random.Random(30).shuffle(all_ids)
    
    #print(all_enc_sents[0:5])
    #print(all_sents[0:5])
    
    batches = len(mult_vids)/batch_size
    batches = int(batches)
        
    i = 0
    j = 0
    
    for n in range(0,batches):
        if j not in vid_batch:
            vid_batch[j] = []
            sent_batch[j] = []
            enc_sent_batch[j]=[]
            cap_len_batch[j] = []

            
        vid_batch[j] = mult_vids[i:i+batch_size]
        sent_batch[j] = all_sents[i:i+batch_size]
        enc_sent_batch[j] = all_enc_sents[i:i+batch_size]
        cap_len_batch[j] = all_cap_len[i:i+batch_size]
        id_batch[j] = all_ids[i:i+batch_size]

        print('parsed batch %d' %j)
        i = i+batch_size
        j = j+1
            
    return vid_batch, sent_batch, enc_sent_batch, cap_len_batch, batches, id_batch   

def extract_sentences(filename):
    
    sentence_set = {}
    
    with open(filename, 'r') as f:
        datastore = json.load(f)
        
    i = 0
    for data in datastore:
        
        #### Extracting only a single sentence per video into a standalone dict

        sentences = data["caption"]
        sentences = [word.lower() for word in sentences] #Normalize the case
        table = str.maketrans('', '', string.punctuation) #Normalize the punctuation
        sentences = [word.translate(table) for word in sentences]

        sentence_set[i] = sentences #0 for only the first sentence\
        
        i = i+1
        
    return sentence_set


def listVocab(sentence_set):
    
    PAD_token = 0
    BOS_token = 1
    EOS_token = 2
    UNK_token = 3
    
    all_tokens = []
    word_count = {}
    token2index = {"<PAD>": 0,"<BOS>":1,"<EOS>":2,"<UNK>":3}
    index2token = {PAD_token: "<PAD>", BOS_token: "<BOS>", EOS_token: "<EOS>", UNK_token: "<UNK>"}
    
    for set_i in sentence_set:
        sentence_set_i = sentence_set[set_i]
        for line in sentence_set_i:
#             line = sentence_set[n]
            tokenized_captions = tokenize(line) #Seperate the words
            all_tokens += tokenized_captions
    
    counter = count_tokens(all_tokens) #Count the word repeatitions in each set
    
    counter_dict = counter.items()
    counter_sort = sorted(counter_dict, key=lambda x:x[1],reverse=True) #sort by frequency of occurance 
    #print(counter_sort)

    i = len(index2token)
    values = [0,1,2,3]
    tokens = ["<PAD>","<BOS>","<EOS>","<UNK>"]
    
    for token, freq in counter_sort:
        word_count[token] = freq
        index2token[i] = token
        token2index[token] = i
        values += [i]
        tokens += [token]
        i+=1
        
    word_count['<PAD>'] = i
    word_count['<BOS>'] = i
    word_count['<EOS>'] = i
    word_count['<UNK>'] = i
    
    bias_init_vector = np.array([1.0 * word_count[ index2token[i] ] for i in index2token])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    
    return [word_count, tokens, values, token2index, index2token, len(index2token),bias_init_vector]
        
        
        
def flattenList(nestedList,output): 
    for i in nestedList: 
        if type(i) == list: 
            flattenList(i,output) 
        else: 
            output.append(i) 
            
    return output

def num_encode(test_sentence,index2token,tokens,tokenized_sentence=[],num_encoded_sentence=[]):
    
    tokenized_sentence.clear()
    num_encoded_sentence.clear()
    
    tokenized_sentence = ["<BOS>"] + tokenize(test_sentence) + ["<EOS>"]
    #print(tokenized_sentence)
    output=[]
    tokenized_sentence = flattenList(tokenized_sentence,output)
    
    cap_len = len(tokenized_sentence)
    
    while len(tokenized_sentence) < MAX_WORDS:
        tokenized_sentence.append("<PAD>")    
    
    #print(len(tokenized_sentence))
    
    for ind, token in enumerate(tokenized_sentence):
        if token in tokens:
            for i in range(0,len(index2token)):
                if token == index2token[i]: 
                    num_encoded_sentence.append(i) 
                    
            #print("token exists")
        else:
            num_encoded_sentence.append(3)
            tokenized_sentence[ind] = tokens[3]
            #print("token unknown")
            
            
                
    #print(len(num_encoded_sentence))

        
    return tokenized_sentence, num_encoded_sentence, cap_len

def schedule_sampling(sampling_prob, cap_len_batch):

        sampling = np.ones(max_caption_len, dtype = bool)
        for l in range(max_caption_len):
            if np.random.uniform(0,1,1) < sampling_prob:
                sampling[l] = True
            else:
                sampling[l] = False
         
        sampling[0] = True
        return sampling
        
def inv_sigmoid(num_epo):

    # 0.88 to 0.12 (-2.0 to 2.0)
    x = np.arange(-2.0, 2.0, (4.0/num_epo))
    y = 1/(1 + np.e**x)
    #y = np.ones(num_epo)
    #print(y)
    return y

def pred_print(pred, cap_len, label, idx2word, batch_size, id_batch):
    
    print_this = np.random.randint(batch_size,size=(1, 10))
    seq=[]
    for i in range(0,batch_size):
        eos_pred = max_caption_len - 1
        eos = cap_len[i] - 1
        for j in range(0, max_caption_len):
                if pred[i][j] == special_tokens['<EOS>']:
                    eos_pred = j
                    break
        myid = id_batch[i]
        pre = list( map (lambda x: idx2word[x] , pred[i][0:eos_pred])  )
        lab = list( map (lambda x: idx2word[x] , label[i][0:eos])  )
        
        pre_no_eos = list( map (lambda x: idx2word[x] , pred[i][0:(eos_pred)])  )
        sen = ' '.join([w for w in pre_no_eos])
        seq.append(sen)
        if i in print_this:      
            print('\nid: ' + str(myid) + '\nanswer: ' + str(lab) + '\nprediction: ' + str(pre))
            
    return seq

def main(_):
    
    output_filename = args.output_filename
    filename_test = args.testing_label
    feat_filepath_test = args.test_feat_filepath
    
    #### PARSE TRAINING DATA #####

    # Extracting captions for each video
    sentence_set = extract_sentences(filename_train)

    word_count, tokens, values, token2index, index2token, n_words, bias_init_vector = listVocab(sentence_set)
    print("There are %d unique words in the captions dataset" % n_words)        
    vocab_num = n_words

    #### PARSE TRAINING DATA #####

    # Extracting captions for each video
    sentence_set = extract_sentences(filename_test)

    mult_vids_test, all_sents_test, all_enc_sents_test, all_cap_len_test, all_ids_test = parse_data_into_lists(filename_test,\
                                                                                      batch_size_test,\
                                                                                  feat_filepath_test,\
                                                                                  index2token,\
                                                                                  tokens)
    
    vid_batch_test, sent_batch_test, intencode_batch_test, cap_len_batch_test, n_batches_test, id_batch_test = \
    parse_data_into_batches(filename_test, batch_size_test, feat_filepath_test, index2token, tokens, mult_vids_test,\
                            all_sents_test, all_enc_sents_test,  all_cap_len_test, all_ids_test)
    
    

    tf.reset_default_graph()

    train_graph = tf.Graph()
    val_graph = tf.Graph()

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    print('train_graph: start')



    with train_graph.as_default():
        feat = tf.placeholder(tf.float32, [None, n_frames, n_inputs], name='video_features')
        captions = tf.placeholder(tf.int32, [None, max_caption_len], name='captions')
        sampling = tf.placeholder(tf.bool, [max_caption_len], name='sampling')
        cap_len = tf.placeholder(tf.int32, [None], name='cap_len')
        model = S2VT(vocab_num=vocab_num, lr=learning_rate)
        logits, loss_op, summary = model.build_model(feat, captions, cap_len, sampling, phases['train'])
        dec_pred = model.inference(logits)
        train_op = model.optimize(loss_op)
        saver = tf.train.Saver(max_to_keep=3)

        init = tf.global_variables_initializer()
    train_sess = tf.Session(graph=train_graph, config=gpu_config)


    with val_graph.as_default():
        feat_val = tf.placeholder(tf.float32, [None, n_frames, n_inputs], name='video_features')
        captions_val = tf.placeholder(tf.int32, [None, max_caption_len], name='captions')
        cap_len_val = tf.placeholder(tf.int32, [None], name='cap_len')

        model_val = S2VT(vocab_num=vocab_num, lr=learning_rate)
        logits_val, loss_op_val, summary_val = model_val.build_model(feat_val, 
                    captions_val, cap_len_val, phase=phases['val'])
        dec_pred_val = model_val.inference(logits_val)

        val_saver = tf.train.Saver(max_to_keep=3)

    val_sess = tf.Session(graph=val_graph, config=gpu_config)
    
    print('saver path: ' + saver_path)
    latest_checkpoint = tf.train.latest_checkpoint(saver_path)

    val_saver.restore(val_sess, latest_checkpoint)

    epo_loss_val = 0
    txt = open(output_filename, 'w')


    for j in range(0,n_batches_test):
        data_batch_val = np.array(vid_batch_test[j])
        label_batch_val = np.array(intencode_batch_test[j])
        id_batch_val = id_batch_test[j]
        caption_lens_batch_val = np.array(cap_len_batch_test[j])

        p_val, summ = val_sess.run([dec_pred_val, summary_val], 
                                    feed_dict={feat_val: data_batch_val,
                                               captions_val: label_batch_val,
                                               cap_len_val: caption_lens_batch_val})


        seq_val = pred_print(p_val, caption_lens_batch_val, label_batch_val, index2token, batch_size_test, id_batch_val)

        for k in range(0, batch_size_test):
                txt.write(id_batch_val[k] + "," + seq_val[k] + "\n")

    print('\nSave file: ' + output_filename)
    txt.close()

#     from subprocess import call

#     call(['python3', 'MLDS_hw2_1_data/bleu_eval.py', output_filename])

    print("Validation: " + str((j+1) * batch_size_test) + "/" + str(n_batches_test) + ", done...")
        
       
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-l', '--testing_label', type=str, default='MLDS_hw2_1_data/testing_label.json')
    parser.add_argument('-t', '--test_feat_filepath', type=str, default="MLDS_hw2_1_data/testing_data/feat/{}.npy")
    parser.add_argument('-o', '--output_filename', type=str, default='testset_output.txt')
    
    args = parser.parse_args()
    
    tf.app.run()        