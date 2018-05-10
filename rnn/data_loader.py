import os

import random
from skimage import io
import torch
import time
from torchvision import transforms


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
        images_list = torch.empty((12 * self.batch_size, 3, 224, 224))
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
                image = io.imread(os.path.join(tmp_path, im))
                image = image.transpose((2, 0, 1))
                # normalize
                m1, m2, m3 = image[0].mean(), image[1].mean(), image[2].mean()
                s1, s2, s3 = image[0].std(), image[1].std(), image[2].std()
                image[0] = (image[0] - m1) / s1
                image[1] = (image[1] - m2) / s2
                image[2] = (image[2] - m3) / s3

                image = torch.from_numpy(image)
                images_list[j * 12 + i] = image
                i += 1
                if (i == 12):
                    break
        return images_list, torch.tensor(labels)
