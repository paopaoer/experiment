import logging
import os
import torch
import time
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import torch.nn as nn
import shutil
import resnet as resnet

batch_size = 10
step_size = 7
num_epochs = 30
momentum = 0.9
gamma = 0.1
learning_rate = 0.0001
weight_decay = 0.005


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

shutil.copy('main.py', results_path + experiment_name + '/main.py', )
shutil.copy('train.py', results_path + experiment_name + '/train.py')
shutil.copy('resnet.py', results_path + experiment_name + '/resnet.py')



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
    if name in ['layer3', 'layer4', 'fc1','lstm']:
        for param in module.parameters():
            param.requires_grad = True
            train_params.append(param)


model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(train_params, lr=learning_rate, weight_decay=weight_decay)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
