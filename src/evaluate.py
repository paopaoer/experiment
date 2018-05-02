import logging
import time
import torch
import copy
import torch.nn as nn
import matplotlib.pyplot as plt
import data_loader as dl


def train_model(model, model_name, optimizer, scheduler, device, num_epochs=25):
    since = time.time()

    data_loader = dl.DataLoader('d:/mydatas/modelnet10', 'val')
    data_size = data_loader.dataset_sizes['val']
    best_model_wts = copy.deepcopy(model.state_dict())

    count = 0

    loss_list = []
    acc_list = []
    x = range(num_epochs)

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)

        model.eval()

        running_loss = 0.0
        running_corrects = 0

        while count >= data_size:

            inputs, label, flag = data_loader.load_data()
            if flag == 0:
                continue
            count += 1
            inputs = inputs.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, prediction = torch.max(outputs, 1)
                loss = nn.CrossEntropyLoss(outputs, label)

            # statistics
            running_loss += loss.item()
            running_corrects += torch.sum(prediction == label.data)

        epoch_loss = running_loss / data_size
        epoch_acc = running_corrects.double() / data_size
        loss_list.append(epoch_loss)
        acc_list.append(epoch_loss)

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, model_name + '.pkl')

        plt.figure()
        plt.xlabel('epoch')
        plt.ylabel('loss/acc')
        plt.title(model_name, fontsize=20)
        loss_line = plt.plot(x, loss_list, color='red', linewidth=1.0, linestyle='--')
        acc_line = plt.plot(x, acc_list, color='blue', linewidth=1.0, linestyle='-')
        plt.legend(handles=[loss_line, acc_line], labels=['loss_line', 'acc_line'], loc='best')
        plt.show()
        plt.savefig(model_name, format('svg'))

        print('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        logging.info('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    logging.info('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    torch.save(model, model_name + '.pkl')
    return model
