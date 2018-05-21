
import time
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import torch.nn as nn
import shutil
import logging
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = 'd:/mydatas/modelnet10'
results_path = '../results/'



batch_size = 10
step_size = 7
num_epochs = 30
momentum = 0.9
gamma = 0.1
learning_rate = 0.0001
weight_decay = 0.005

num_img = 12
num_classes = 10