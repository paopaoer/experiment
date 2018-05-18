import os
import logging
import time
import copy
import torch

import torch.nn as nn
import matplotlib.pyplot as plt
import data_loader as dl


def train_model(model, exe_name, optimizer, scheduler, num_epochs, batch_size):
    since = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    loss_list = {'train': [], 'val': []}
    acc_list = {'train': [], 'val': []}

    path = '../result/' + exe_name + '/'

    x = range(num_epochs)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            data_loader = dl.DataLoader('f:/json/alexnet2', phase, batch_size)
            data_size = data_loader.dataset_sizes[phase]
            count = {'train': 0, 'val': 0}

            while count[phase] < data_size:

                inputs, labels = data_loader.load_data2()
                count[phase] += batch_size
                print(count[phase], end=' ', )

                inputs = inputs.view(batch_size, 12,-1)

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, prediction = torch.max(outputs, 1)
                    # print('labels: ',labels)
                    # print('prediction: ',prediction)

                    criterion = nn.CrossEntropyLoss()

                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                # statistics
                running_loss += loss.item() * labels.size(0)
                running_corrects += torch.sum(prediction == labels.data)

                print('{} Loss: {:.4f} '.format(phase, loss.item()))

            epoch_loss = running_loss / data_size
            epoch_acc = running_corrects.double() / data_size
            loss_list[phase].append(epoch_loss)
            acc_list[phase].append(epoch_acc)

            print('{} Loss: {:.6f} Acc: {:.6f}'.format(phase, epoch_loss, epoch_acc))
            logging.info('{} Loss: {:.6f} Acc: {:.6f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, path + exe_name + '.pkl')
        print()

    # draw acc
    plt.figure(1)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title(exe_name + '_acc')
    train_acc_line, = plt.plot(x, acc_list['train'], color='red', linewidth=1.0, linestyle='--')
    val_acc_line, = plt.plot(x, acc_list['val'], color='blue', linewidth=1.0, linestyle='-')
    plt.legend(handles=[train_acc_line, val_acc_line], labels=['train_acc_line', 'val_acc_line'], loc='best')
    plt.savefig(path + exe_name + '_acc')

    # draw loss
    plt.figure(2)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(exe_name + '_loss')
    train_loss_line, = plt.plot(x, loss_list['train'], color='red', linewidth=1.0, linestyle='--')
    val_loss_line, = plt.plot(x, loss_list['val'], color='blue', linewidth=1.0, linestyle='-')
    plt.legend(handles=[train_loss_line, val_loss_line], labels=['train_loss_line', 'val_loss_line'], loc='best')
    plt.savefig(path + exe_name + '_loss')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best {} Acc: {:6f}'.format(phase, best_acc))
    logging.info('Best {} Acc: {:6f}'.format(phase, best_acc))

    model.load_state_dict(best_model_wts)
    torch.save(model, path + exe_name + '.pkl')

    return model
