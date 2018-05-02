import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import logging


def train_model(model, model_name, optimizer, scheduler, device, num_epochs=30):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    loss_list = {'train': [], 'val': []}
    acc_list = {'train': [], 'val': []}

    x = range(num_epochs)

    path = 'result/' + model_name + '/'

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    criterion = nn.CrossEntropyLoss()
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                print('{} Loss: {:.4f} '.format(phase,loss.item()))
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            loss_list[phase].append(epoch_loss)
            acc_list[phase].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, path + model_name + '.pkl')

            # draw acc
        plt.figure(1)
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.title(model_name + '_acc')
        train_acc_line, = plt.plot(x, acc_list['train'], color='red', linewidth=1.0, linestyle='--')
        val_acc_line, = plt.plot(x, acc_list['val'], color='blue', linewidth=1.0, linestyle='-')
        plt.legend(handles=[train_acc_line, val_acc_line], labels=['train_acc_line', 'val_acc_line'], loc='best')
        plt.savefig(path + model_name + '_acc')
        plt.show()

        # draw loss
        plt.figure(2)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title(model_name + '_loss')
        train_loss_line, = plt.plot(x, loss_list['train'], color='red', linewidth=1.0, linestyle='--')
        val_loss_line, = plt.plot(x, loss_list['val'], color='blue', linewidth=1.0, linestyle='-')
        plt.legend(handles=[train_loss_line, val_loss_line], labels=['train_loss_line', 'val_loss_line'], loc='best')
        plt.savefig(path + model_name + '_loss')
        plt.show()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best {} Acc: {:4f}'.format(phase, best_acc))
    logging.info('Best {} Acc: {:4f}'.format(phase, best_acc))

    model.load_state_dict(best_model_wts)
    torch.save(model, path + model_name + '.pkl')

    return model


if __name__ == '__main__':
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'd:/mydatas/single_view/06'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = models.resnet18(pretrained=False)

    model_ft.fc = nn.Linear(512, 10)

    model_ft = model_ft.to(device)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model_ft.parameters())

    # Decay LR by a factor of 0.1 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

    model_name = 'single_view_adam'
    if not os.path.exists('result/' + model_name):
        os.makedirs('result/' + model_name)

    log = logging.basicConfig(level=logging.INFO,
                              format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                              datefmt='%a, %d %b %Y %H:%M:%S',
                              filename='result/' + model_name + '/' + model_name + '.log', filemode='a')

    model_ft = train_model(model_ft, model_name, optimizer_ft, exp_lr_scheduler, device, num_epochs=30)
