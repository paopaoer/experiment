import logging
import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models

import  torch.nn as nn

import model.all_view as av
import model.mvcnn  as mvcnn
import model.multi_channel as mc
import train

if __name__ == '__main__':
    model_name = 'all_view_batch4_pre_weight'
    if not os.path.exists('result/' + model_name):
        os.makedirs('result/' + model_name)

    log = logging.basicConfig(level=logging.INFO,
                              format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                              datefmt='%a, %d %b %Y %H:%M:%S',
                              filename='result/' + model_name + '/' + model_name + '.log', filemode='a')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    m = models.resnet18(pretrained=True)

    print(m.state_dict().keys())

    my_model = av.resnet18()

    pretrained_dict = m.state_dict()
    model_dict =my_model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    my_model.load_state_dict(model_dict)


    for name, module in my_model.named_children():
        if name in ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']:
            print(name, module)
            for param in module.parameters():
                param.requires_grad = False

    train_params = []

    for name, module in my_model.named_children():
        if name in ['layer4', 'bn2', 'fc1', 'fc2']:
            print(name, module)
            for param in module.parameters():
                param.requires_grad = True
                train_params.append(param)

    # Parameters of newly constructed modules have requires_grad=True by default

    my_model = my_model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opoosed to before.
    my_optimizer = optim.SGD(train_params, lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(my_optimizer, step_size=4, gamma=0.1)




    train.train_model(my_model, model_name, my_optimizer, exp_lr_scheduler, device, num_epochs=30)


