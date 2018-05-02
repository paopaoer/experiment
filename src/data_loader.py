import os

import random
from skimage import io, transform
import torch

import numpy as  np


class DataLoader:
    data_sizes = {'train': 800, 'val': 200}

    def __init__(self, root, dir):

        self.root = root
        self.path = os.path.join(root, dir)
        self.class_names = os.listdir(self.path)
        self.class_names.sort()
        self.class_to_idx = {self.class_names[i]: i for i in range(len(self.class_names))}
        self.iters_list = []

        for cn in self.class_names:
            files_list = os.listdir(os.path.join(self.path, cn))
            files_list.sort()
            tmp_iter = iter(files_list)
            self.iters_list.append(tmp_iter)

    def load_data(self):
        number = random.randint(0, 9)
        class_name = self.class_names[number]
        label = self.class_to_idx[class_name]
        tmp_path = os.path.join(self.path, class_name)
        print(class_name)
        print('label: ', label)
        i = 0
        flag = 0
        images_list = torch.empty((12, 3, 224, 224))
        for im in self.iters_list[label]:
            image = io.imread(os.path.join(tmp_path, im))
            image = image.transpose((2, 0, 1))
            image = torch.from_numpy(image)
            images_list[i] = image
            i += 1
            flag = 1
            if (i == 12):
                break
        return images_list, torch.LongTensor(label), flag
