import copy
import time
import matplotlib.pyplot as plt

from main import *
import DataLoader as DL

since = time.time()
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

loss_list = {'train': [], 'val': []}
acc_list = {'train': [], 'val': []}

path = results_path + experiment_name + '/'

x = range(num_epochs)

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
    logging.info('-' * 10)

    count = {'train': 0, 'val': 0}
    data_loader = DL.DataLoader(data_dir)
    classes_name = data_loader.class_names

    for phase in ['train', 'val']:
        if phase == 'train':
            exp_lr_scheduler.step()
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        classes_acc = []
        classes_total = []
        for i in range(num_classes):
            classes_acc.append(0)
        for i in range(num_classes):
            classes_total.append(0)

        data_size = data_loader.dataset_sizes[phase]

        while count[phase] < data_size:

            inputs, labels = data_loader.load_data(phase)

            count[phase] += batch_size
            print(count[phase], end=' ')
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, prediction = torch.max(outputs, 1)

                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            print('{} Loss: {:.4f} '.format(phase, loss.item()))

            # statistics
            running_loss += loss.item() * labels.size(0)
            running_corrects += torch.sum(prediction == labels.data)

            for pre, label in zip(prediction, labels):
                classes_total[label] += 1
                if pre == label.data:
                    classes_acc[label] += 1
        # print(classes_total)

        for i in range(num_classes):
            class_acc = classes_acc[i] / classes_total[i]
            print('{} Acc: {:.4f}'.format(classes_name[i], class_acc))
            logging.info('{} Acc: {:.4f}'.format(classes_name[i], class_acc))

        epoch_loss = running_loss / data_size
        epoch_acc = running_corrects.double() / data_size
        loss_list[phase].append(epoch_loss)
        acc_list[phase].append(epoch_acc)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, path + experiment_name + '.pkl')
    print()

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best {} Acc: {:4f}'.format(phase, best_acc))
logging.info('Best {} Acc: {:4f}'.format(phase, best_acc))

model.load_state_dict(best_model_wts)
torch.save(model, path + experiment_name + '.pkl')

# draw acc
plt.figure(1)
plt.xlabel('epoch')
plt.ylabel('acc')
plt.title(experiment_name + '_acc')
train_acc_line, = plt.plot(x, acc_list['train'], color='red', linewidth=1.0, linestyle='--')
val_acc_line, = plt.plot(x, acc_list['val'], color='blue', linewidth=1.0, linestyle='-')
plt.legend(handles=[train_acc_line, val_acc_line], labels=['train_acc_line', 'val_acc_line'], loc='best')
plt.savefig(path + experiment_name + '_acc')
plt.show()

# draw loss
plt.figure(2)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title(experiment_name + '_loss')
train_loss_line, = plt.plot(x, loss_list['train'], color='red', linewidth=1.0, linestyle='--')
val_loss_line, = plt.plot(x, loss_list['val'], color='blue', linewidth=1.0, linestyle='-')
plt.legend(handles=[train_loss_line, val_loss_line], labels=['train_loss_line', 'val_loss_line'], loc='best')
plt.savefig(path + experiment_name + '_loss')
plt.show()
