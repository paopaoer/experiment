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
input_size = 512
hidden_size = 256
num_layers = 2
num_classes = 10
batch_size = 1
num_epochs = 2
learning_rate = 0.001

if __name__ == '__main__':
    exe_name = 'rnn'
    if not os.path.exists('../result/' + exe_name):
        os.makedirs('../result/' + exe_name)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename='../result/' + exe_name + '/' + exe_name + '.log', filemode='a')



    pretrained_model=torch.load('mpdelnet10_no_pre.pkl',map_location='cpu')
    #print(pretrained_model.state_dict().keys())
    resnet = resnet.resnet18(pretrained=False)
    pretrained_dict = pretrained_model.state_dict()
    model_dict = resnet.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    resnet.load_state_dict(model_dict)




    rnn = rnn.RNN(input_size, hidden_size, num_layers, num_classes)
    rnn_optimizer = optim.SGD(rnn.parameters(), lr=learning_rate, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(rnn_optimizer, step_size=5, gamma=0.1)
    train.train_model(rnn, resnet,exe_name, rnn_optimizer, exp_lr_scheduler, num_epochs,batch_size)
