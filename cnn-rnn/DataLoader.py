import os

import random
from skimage import io
import torch

from main import batch_size, num_img


class DataLoader:

    def __init__(self, root):
        self.root = root
        self.class_names = os.listdir(os.path.join(root, 'train'))
        self.class_names.sort()
        self.class_to_idx = {self.class_names[i]: i for i in range(len(self.class_names))}

        train_count = {}
        val_count = {}
        train_total = 0
        val_total = 0
        self.train_iters_list = []
        self.val_iters_list = []
        train_shuffle_array = []
        val_shuffle_array = []
        train_path = os.path.join(root, 'train')
        val_path = os.path.join(root, 'val')

        # count
        for cn in self.class_names:
            files_list = os.listdir(os.path.join(train_path, cn))
            files_list.sort()
            train_count[cn] = (len(files_list)) / num_img
            for _ in range(int(train_count[cn])):
                train_shuffle_array.append(self.class_to_idx[cn])
            train_total += train_count[cn]
            tmp_iter = iter(files_list)
            self.train_iters_list.append(tmp_iter)

        for cn in self.class_names:
            files_list = os.listdir(os.path.join(val_path, cn))
            files_list.sort()
            val_count[cn] = (len(files_list)) / num_img
            for _ in range(int(val_count[cn])):
                val_shuffle_array.append(self.class_to_idx[cn])
            val_total += val_count[cn]
            tmp_iter = iter(files_list)
            self.val_iters_list.append(tmp_iter)

        self.dataset_sizes = {'train': train_total, 'val': val_total}

        random.shuffle(train_shuffle_array)


        random.shuffle(val_shuffle_array)
        self.train_shuffle_iter = iter(train_shuffle_array)
        self.val_shuffle_iter = iter(val_shuffle_array)

    def load_data(self, phase):

        labels = []
        images_list = torch.empty((num_img * batch_size, 3, 224, 224))
        if phase == 'train':
            shuffel_iter = self.train_shuffle_iter
            iters_list = self.train_iters_list
        else:
            shuffel_iter = self.val_shuffle_iter
            iters_list = self.val_iters_list

        for j in range(batch_size):

            number = next(shuffel_iter)
            class_name = self.class_names[number]
            label = self.class_to_idx[class_name]
            labels.append(label)
            path = os.path.join(self.root, phase)
            tmp_path = os.path.join(path, class_name)
            i = 0
            for im in iters_list[label]:
                image = io.imread(os.path.join(tmp_path, im))
                image = image.transpose((2, 0, 1))
                # normalize
                m = [0.485, 0.456, 0.406]
                s = [0.229, 0.224, 0.225]

                image[0] = (image[0] - m[0]) / s[0]
                image[1] = (image[1] - m[1]) / s[1]
                image[2] = (image[2] - m[2]) / s[2]

                image = torch.from_numpy(image)
                images_list[j * num_img + i] = image
                i += 1
                if (i == num_img):
                    break
        return images_list, torch.tensor(labels)
