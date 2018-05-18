import os

import random
from skimage import io,transform,color
import torch
import time
from torchvision import transforms
import json

class DataLoader:

    def __init__(self, root, phase, batch_size):

        self.batch_size = batch_size
        self.dataset_sizes = {'train': 800, 'val': 200}
        self.root = root
        self.phase = phase
        self.path = os.path.join(root, phase)
        self.class_names = os.listdir(self.path)
        self.class_names.sort()
        self.class_to_idx = {self.class_names[i]: i for i in range(len(self.class_names))}
        self.iters_list = []

        train_shuffle_array = []
        for i in range(self.dataset_sizes['train']):
            train_shuffle_array.append((i % 10))
        random.shuffle(train_shuffle_array)

        val_shuffle_array = []
        for i in range(self.dataset_sizes['val']):
            val_shuffle_array.append((i % 10))
        random.shuffle(val_shuffle_array)
        self.train_shuffle_iter = iter(train_shuffle_array)
        self.val_shuffle_iter = iter(val_shuffle_array)

        for cn in self.class_names:
            files_list = os.listdir(os.path.join(self.path, cn))
            files_list.sort()
            tmp_iter = iter(files_list)
            self.iters_list.append(tmp_iter)

    def load_data(self):

        labels = []
        images_list = torch.empty((12 * self.batch_size, 28,28))
        if self.phase == 'train':
            iter_array = self.train_shuffle_iter
        else:
            iter_array = self.val_shuffle_iter

        for j in range(self.batch_size):

            number = next(iter_array)
            class_name = self.class_names[number]
            label = self.class_to_idx[class_name]
            labels.append(label)
            tmp_path = os.path.join(self.path, class_name)
            i = 0
            for im in self.iters_list[label]:
                image = io.imread(os.path.join(tmp_path, im),as_grey=True)
                #image=color.rgb2gray(image)

                image=transform.resize(image,(28,28))
                #image = image.transpose((2, 0, 1))
                # normalize
                m1 = image.mean()
                s1 = image.std()
                image= (image - m1) / s1
            

                image = torch.from_numpy(image)
    
                images_list[j * 12 + i] = image
                i += 1
                if (i == 12):
                    break
        return images_list, torch.tensor(labels)


    def load_data2(self):

        labels = []
        images_list = torch.empty(self.batch_size, 12,512)
        
        if self.phase == 'train':
            iter_array = self.train_shuffle_iter
        else:
            iter_array = self.val_shuffle_iter

        for j in range(self.batch_size):

            number = next(iter_array)
            class_name = self.class_names[number]
            label = self.class_to_idx[class_name]
            labels.append(label)
            tmp_path = os.path.join(self.path, class_name)

            im=next(self.iters_list[label])
    
            with open(os.path.join(tmp_path, im), 'r') as f:
                data = json.load(f)
                #for i in range(12):
                    #data[i]=data[i][::8]
                #print(data)
                data=torch.tensor(data)
                images_list[j]=data

                
        return images_list, torch.tensor(labels)
