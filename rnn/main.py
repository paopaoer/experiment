import logging
import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models

import torch.nn as nn

import resnet
import rnn
import train

sequence_length = 12
input_size =512
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 50
learning_rate = 0.01

if __name__ == '__main__':
    exe_name = 'rnn_alexnet2'
    if not os.path.exists('../result/' + exe_name):
        os.makedirs('../result/' + exe_name)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename='../result/' + exe_name + '/' + exe_name + '.log', filemode='a')


    rnn = rnn.RNN(input_size, hidden_size, num_layers, num_classes)
    rnn_optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)
    exp_lr_scheduler = lr_scheduler.StepLR(rnn_optimizer, step_size=10, gamma=0.1)
    train.train_model(rnn,exe_name, rnn_optimizer, exp_lr_scheduler, num_epochs,batch_size)
