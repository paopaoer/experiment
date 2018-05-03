import logging
import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

import model.all_view as ma
import model.mvcnn  as mvcnn
import model.multi_channel as mc
import train

if __name__ == '__main__':
    model_name = 'all_view'
    if not os.path.exists('result/' + model_name):
        os.makedirs('result/' + model_name)

    log = logging.basicConfig(level=logging.INFO,
                              format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                              datefmt='%a, %d %b %Y %H:%M:%S',
                              filename='result/' + model_name + '/' + model_name + '.log', filemode='a')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model = ma.resnet18(pretrained=False)
    #model = mvcnn.resnet18(pretrained=False)
    model = mc.resnet18(pretrained=False)

    model = model.to(device)

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.Adam(model.parameters())

    # Decay LR by a factor of 0.1 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    model_ft = train.train_model(model, model_name, optimizer, exp_lr_scheduler, device, num_epochs=10)
