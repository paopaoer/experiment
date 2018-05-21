import shutil
import logging
import os

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import torch.nn as nn

import cnn.resnet as resnet
#import cnn.Inception as inception

batch_size = 10
step_size = 7
num_epochs = 30
momentum = 0.9
gamma = 0.1
learning_rate = 0.001
weight_decay = 0.01

num_img = 12
num_classes = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = 'd:/mydatas/modelnet10'
results_path = '../results/'

experiment_name = 'test'
if not os.path.exists(results_path + experiment_name):
    os.makedirs(results_path + experiment_name)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=results_path + experiment_name + '/' + experiment_name + '.log', filemode='a')

shutil.copy('main.py', results_path + experiment_name + '/main.py')

'''
shutil.copy('cnn/Inception.py', results_path + experiment_name + '/Inception.py')
pretrained_model = models.inception_v3(pretrained=True)
print(pretrained_model.state_dict().keys())
pretrained_dict = pretrained_model.state_dict()

model = inception.inception_v3(pretrained=False)
print(model.state_dict().keys())
model_dict = model.state_dict()

pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

train_params = []

for name, module in model.named_children():
    if name in ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3','Conv2d_3b_1x1', 'Conv2d_4a_3x3']:
        for param in module.parameters():
            param.requires_grad = False

for name, module in model.named_children():
    if name in ['Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d', 'Mixed_6e','Mixed_7a','Mixed_7b','Mixed_7c','fc1','AuxLogits1']:
        for param in module.parameters():
            param.requires_grad = True
            train_params.append(param)


'''
shutil.copy('cnn/resnet.py', results_path + experiment_name + '/resnet.py')

pretrained_model = models.resnet18(pretrained=True)
# print(m.state_dict().keys())
pretrained_dict = pretrained_model.state_dict()

model = resnet.resnet18()
model_dict = model.state_dict()

pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

train_params = []

for name, module in model.named_children():
    if name in ['conv1', 'bn1', 'layer1', 'layer2']:
        for param in module.parameters():
            param.requires_grad = False

for name, module in model.named_children():
    if name in ['layer3', 'layer4', 'bn2', 'fc1', 'fc2']:
        for param in module.parameters():
            param.requires_grad = True
            train_params.append(param)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(train_params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
