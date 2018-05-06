import os
import logging
import time
import copy
import torch

import torch.nn as nn
import matplotlib.pyplot as plt
import data_loader as dl


def train_model(model, model_name, optimizer, scheduler, device, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    loss_list = {'train': [], 'val': []}
    acc_list = {'train': [], 'val': []}

    path = 'result/' + model_name + '/'

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
            batch_size = 4
            data_loader = dl.DataLoader('d:/mydatas/modelnet10', phase, batch_size)
            data_size = data_loader.dataset_sizes[phase]
            count = {'train': 0, 'val': 0}

            while count[phase] < data_size:

                inputs, labels, = data_loader.load_data3()
                count[phase] += batch_size
                print(count[phase], end=' ', )
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, prediction = torch.max(outputs, 1)

                    criterion = nn.CrossEntropyLoss()
                    #print('labels: ',labels)
                    #print('prediction: ',prediction)


                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                print('{} Loss: {:.4f} '.format(phase, loss.item()))

                # statistics
                running_loss += loss.item() * (inputs.size(0) / 12)
                running_corrects += torch.sum(prediction == labels.data)

            epoch_loss = running_loss / data_size
            epoch_acc = running_corrects.double() / data_size
            loss_list[phase].append(epoch_loss)
            acc_list[phase].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, path + model_name + '.pkl')
        print()

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

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best {} Acc: {:4f}'.format(phase, best_acc))
    logging.info('Best {} Acc: {:4f}'.format(phase, best_acc))

    model.load_state_dict(best_model_wts)
    torch.save(model, path + model_name + '.pkl')

    return model
