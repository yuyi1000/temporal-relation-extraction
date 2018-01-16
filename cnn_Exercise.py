from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import gzip
import pickle

import numpy
import tensorflow as tf

from train_cnn import ConvNet
from read_file import *



# Set parameters 
parser = argparse.ArgumentParser('CNN Exercise.')
parser.add_argument('--learning_rate', 
                    type=float, default=0.1,
                    help='Initial learning rate.')
parser.add_argument('--num_epochs',
                    type=int,
                    default=5000, 
                    help='Number of epochs to run trainer.')
parser.add_argument('--drop_out',
                    type=float,
                    default=0.4, 
                    help='drop out rate for last layer.')
parser.add_argument('--decay',
                    type=float,
                    default=0.001, 
                    help='Decay rate of l2 regularization.')
parser.add_argument('--batch_size', 
                    type=int, default=100, 
                    help='Batch size. Must divide evenly into the dataset sizes.')
parser.add_argument('--log_dir', 
                    type=str, 
                    default='logs', 
                    help='Directory to put logging.')
parser.add_argument('--visibleSize',
                    type=int,
                    default=str(28 * 28),
                    help='Used for gradient checking.')
parser.add_argument('--hidden_size', 
                    type=int,
                    default='100',
                    help='.')
parser.add_argument('--model_path',
                    type=str,
                    # default='../model/cnn_event_vs_event_ckpt',
                    default='../model/cnn_event_vs_time.ckpt',
                    # default='../model/cnn.ckpt',
                    help='Path of the trained model')
parser.add_argument('--embedding_path', 
                    type=str, 
                    default='../data/embedding_with_xml_tag.pkl',
                    # default='../data/embedding_without_xml_tag.pkl', 
                    help='Path of the pretrained word embedding.')
parser.add_argument('--thyme_data_dir', 
                    type=str, 
                    # default='../data/padding.pkl', 
                    # default='../data/padding_test_event_vs_event.pkl',
                    # default='../data/padding_test_event_vs_time.pkl',
                    # default='../data/padding_event_vs_event.pkl',
                    # default='../data/padding_event_vs_time_with_xml_tag.pkl',
                    default='../data/padding_event_vs_time_without_xml_tag.pkl',
                    # default='../data/padding_event_vs_time_without_xml_tag_pos_embed_source.pkl',
                    help='Directory to put the thyme data.')

 
FLAGS = None
FLAGS, unparsed = parser.parse_known_args()
mode = int(sys.argv[1])


def show_sentence(sent_emb, id_to_word):
  for idx in sent_emb:
    # when idx is 0, it is a padding word.
    if idx == 0:
      break
    print (id_to_word[idx], end=" ")
  print ()



def replace_after_with_before(sent_emb):
  for i in range(len(sent_emb)):
    if sent_emb[i] == 148 or sent_emb[i] == 429:
      sent_emb[i] = 178
  

# get data set for an interested word
# E.g. we may want to investigate the word "after" and its impact
# word_to_id: {"after": 148, "After": 429, "before": 178}
def extract_word(data_set, id_to_word):
  lst = []
  sent_embed = data_set[0]
  pos_source_embed = data_set[1]
  pos_target_embed = data_set[2]
  event_one_hot = data_set[3]
  timex3_one_hot = data_set[4]
  source_one_hot = data_set[5]
  target_one_hot = data_set[6]
  boolean_features = data_set[7]
  label = data_set[8]

  n = len(data_set[0])
  print ("n: ", n)

  idx = 0

  # print ("id_to_word: ", id_to_word)

  
  for i in range(n):
    if label[i] == 1 and (148 in sent_embed[i] or 429 in sent_embed[i]):
      lst.append(i)
      print ("idx: ", idx)
      replace_after_with_before(sent_embed[i])
      show_sentence(sent_embed[i], id_to_word)
      idx += 1
      
  print ("lst: ", lst)
  print ("lst length: ", len(lst))

  filtered_data_set = [[sent_embed[i] for i in lst], [pos_source_embed[i] for i in lst], [pos_target_embed[i] for i in lst], \
                       [event_one_hot[i] for i in lst], [timex3_one_hot[i] for i in lst], [source_one_hot[i] for i in lst], \
                       [target_one_hot[i] for i in lst], [boolean_features[i] for i in lst], [label[i] for i in lst]]

  return filtered_data_set

# add validation dataset to training dataset
def combine_train_and_dev(train_set, dev_set):
  for i in range(len(train_set)):
    train_set[i] = numpy.concatenate((train_set[i], dev_set[i]))



# ======================================================================
#  STEP 0: Load pre-trained word embeddings and the SNLI data set
#

embedding = pickle.load(open(FLAGS.embedding_path, 'rb'))

# snli = pickle.load(open(FLAGS.snli_data_dir, 'rb'))
# train_set = snli[0]
# dev_set   = snli[1]
# test_set  = snli[2]

thyme = pickle.load(open(FLAGS.thyme_data_dir, 'rb'))
train_set = thyme[0]
print("number of instances in training set: ", len(train_set[0]))
dev_set   = thyme[1]
# combine_train_and_dev(train_set, dev_set)
# print("number of instances in combined training set: ", len(train_set[0]))
test_set  = thyme[2]
closure_test_set = thyme[3]
train_label_count = thyme[4]

# test_after_set = extract_word(test_set, embedding.id_to_word)


# ====================================================================
# Use a smaller portion of training examples (e.g. ratio = 0.1) 
# for debuging purposes.
# Set ratio = 1 for training with all training examples.

# ratio = 1

# train_size = train_set[0].shape[0]
# idx = list(range(train_size))
# idx = numpy.asarray(idx, dtype=numpy.int32)

# # Shuffle the train set.
# for _ in range(7):
#   numpy.random.shuffle(idx)

# # Get a certain ratio of the training set.
# idx = idx[0:int(idx.shape[0] * ratio)]
# sent1 = train_set[0][idx]
# leng1 = train_set[1][idx]
# sent2 = train_set[2][idx]
# leng2 = train_set[3][idx]
# label = train_set[4][idx]

# train_set = [sent1, leng1, sent2, leng2, label]

# Use a smaller portion of training examples (e.g. ratio = 0.1) 
# for debuging purposes.
# Set ratio = 1 for training with all training examples.

ratio = 1

train_size = train_set[0].shape[0]
idx = list(range(train_size))
idx = numpy.asarray(idx, dtype=numpy.int32)

# Shuffle the train set.
for _ in range(7):
  numpy.random.shuffle(idx)

# Get a certain ratio of the training set.
idx = idx[0:int(idx.shape[0] * ratio)]
sent_embed = train_set[0][idx]

# pos_embed_source = train_set[1][idx]
# pos_embed_target = train_set[2][idx]

pos_embed_first_entity = train_set[1][idx]
pos_embed_second_entity = train_set[2][idx]

event_bitmap = train_set[3][idx]
timex3_bitmap = train_set[4][idx]
source_bitmap = train_set[5][idx]
target_bitmap = train_set[6][idx]
# boolean_features = train_set[7][idx]
# label = train_set[8][idx]
label = train_set[7][idx]
# label = train_set[5][idx]

# train_set = [sent_embed, pos_embed_source, pos_embed_target, event_bitmap, timex3_bitmap, source_bitmap, target_bitmap, boolean_features, label]
# train_set = [sent_embed, pos_embed_source, pos_embed_target, event_bitmap, timex3_bitmap, source_bitmap, target_bitmap, label]
train_set = [sent_embed, pos_embed_first_entity, pos_embed_second_entity, event_bitmap, timex3_bitmap, source_bitmap, target_bitmap, label]
# train_set = [sent_embed, pos_embed_source, pos_embed_target, event_bitmap, timex3_bitmap, label]

# ======================================================================
#  STEP 1: Train a baseline model.
#  This trains a feed forward neural network with one hidden layer.
#
#  Expected accuracy: 97.80%

if mode == 1:
  cnn = ConvNet(1)
  accuracy = cnn.train_and_evaluate(FLAGS, embedding, train_set, dev_set, test_set, train_label_count)

  # Output accuracy.
  print(20 * '*' + 'model 1' + 20 * '*')
  print('accuracy is %f' % (accuracy))
  print()


# ======================================================================
#  STEP 2: Use one convolutional layer.
#  
#  Expected accuracy: 98.78%

if mode == 2:
  cnn = ConvNet(2)
  accuracy = cnn.train_and_evaluate(FLAGS, embedding, train_set, dev_set, test_set, closure_test_set, train_label_count)
  # accuracy = cnn.train_and_evaluate(FLAGS, embedding, train_set, dev_set, test_after_set)
  # Output accuracy.
  print(20 * '*' + 'model 2' + 20 * '*')
  print('accuracy is %f' % (accuracy))
  print()
