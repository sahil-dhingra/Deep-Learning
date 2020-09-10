from cs231n.rnn_layers import *
from cs231n.classifiers.rnn import CaptioningRNN
from cs231n.captioning_solver import CaptioningSolver

import time, os, json
import numpy as np
import matplotlib.pyplot as plt
import nltk

from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.rnn_layers import *
from cs231n.captioning_solver import CaptioningSolver
from cs231n.classifiers.rnn import CaptioningRNN
from cs231n.coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from cs231n.image_utils import image_from_url


class MyCaptioningModel(object):
    
    def __init__(self, data, max_train_data=2000, pca=True, hidden_dim=512, wordvec_dim=256, num_epochs=50, batch_size=50, lr=5e-3, lr_decay=0.997):
        
        np.random.seed(231)

        self.small_data = load_coco_data(max_train=max_train_data, pca_features=pca)

        self.small_lstm_model = CaptioningRNN(
                  cell_type='lstm',
                  word_to_idx=data['word_to_idx'],
                  input_dim=data['train_features'].shape[1],
                  hidden_dim=hidden_dim,
                  wordvec_dim=wordvec_dim,
                  dtype=np.float32,
                )
        
        self.small_lstm_solver = CaptioningSolver(self.small_lstm_model, self.small_data,
                   update_rule='adam',
                   num_epochs=50,
                   batch_size=50,
                   optim_config={
                     'learning_rate': lr,
                   },
                   lr_decay=lr_decay,
                   verbose=True, print_every=10,
                 )
        
    def train(self):
        self.small_lstm_solver.train()
    