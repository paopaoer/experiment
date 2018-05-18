import cv2
import numpy as np

import os
import re
import shutil

src_dir = '/home/lss/ModelNet10/ModelNet10'
train_dst_dir = '/home/lss/PycharmProjects/cnn/test/99/train'
val_dst_dir = '/home/lss/PycharmProjects/cnn/test/99/val'


def create_dirs(src_dir):

    sub_dirs = os.listdir(src_dir)
    print(sub_dirs)
    for sd in sub_dirs:
        if sd.find('.DS Store') > -1:
            continue
        if not os.path.exists(train_dst_dir+'/'+sd):
            os.mkdir(train_dst_dir+'/'+sd)
        if not os.path.exists(val_dst_dir+'/'+sd):
            os.mkdir(val_dst_dir+'/'+sd)
    return sub_dirs


'''
def copy_files(src_dir):
    sub_dirs = create_dirs(src_dir)
    for sd in sub_dirs:
        train_list_files = src_dir+'/'+sd+'/'+'train'
        for tlf in os.listdir(train_list_files):
            if (tlf.find('.off'))>-1:
                continue
            if(tlf.find('x0_y0_z1'))>-1:
                dest_file=train_dst_dir+'/'+sd+'/'+tlf
                if not os.path.exists(dest_file):
                    shutil.copy(train_list_files+'/'+tlf,dest_file)
        val_list_files = src_dir+'/'+sd+'/'+'test'
        for vlf in os.listdir(val_list_files):
            if(tlf.find('.off'))>-1 :
                continue
            if(vlf.find('x0_y0_z1'))>-1:
                dest_file=val_dst_dir+'/'+sd+'/'+vlf
                if not os.path.exists(dest_file):
                    shutil.copy(val_list_files+'/'+vlf,dest_file)


'''


def com_ima(path, sd, phase):
    k = 0

    m1 = np.zeros((256, 256, 4), np.uint8)
    m2 = np.zeros((256, 256, 4), np.uint8)
    m3 = np.zeros((256, 256, 4), np.uint8)
    m4 = np.zeros((256, 256, 4), np.uint8)

    list_dirs = os.listdir(path)
    list_dirs.sort()

    for tlf in list_dirs:

        if (tlf.find('.off')) > -1:
            continue
        if(tlf.find('x1_y1_z1')) > -1:
            m1 = cv2.imread(path+'/'+tlf, cv2.IMREAD_UNCHANGED)
        if(tlf.find('x1_y1_z-1')) > -1:
            m2 = cv2.imread(path+'/'+tlf, cv2.IMREAD_UNCHANGED)
        if(tlf.find('x-1_y1_z1')) > -1:
            m3 = cv2.imread(path+'/'+tlf, cv2.IMREAD_UNCHANGED)
        if(tlf.find('x-1_y1_z-1')) > -1:
            m4 = cv2.imread(path+'/'+tlf, cv2.IMREAD_UNCHANGED)

        k += 1

        if k == 14:

            empty_image = np.zeros((512, 512, 4), np.uint8)

            for i in range(256):
                for j in range(256):
                    empty_image[i][j] = m1[i][j]
            for i in range(256, 512):
                for j in range(256):
                    empty_image[i][j] = m2[i-256][j]
            for i in range(256):
                for j in range(256, 512):
                    empty_image[i][j] = m3[i][j-256]
            for i in range(256, 512):
                for j in range(256, 512):
                    empty_image[i][j] = m4[i-256][j-256]

            index = tlf.find('x')
            image_name = tlf[:index]+'.png'
            if(phase == 'train'):
                dest_file = train_dst_dir+'/'+sd+'/'+image_name
            else:
                dest_file = val_dst_dir+'/'+sd+'/'+image_name
            print (dest_file)
            if not os.path.exists(dest_file):
                cv2.imwrite(dest_file, empty_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

            k = 0


def copy_files(src_dir):
    sub_dirs = create_dirs(src_dir)
    for sd in sub_dirs:
        train_list_files = src_dir+'/'+sd+'/'+'train'
        com_ima(train_list_files, sd, 'train')

        val_list_files = src_dir+'/'+sd+'/'+'test'
        com_ima(val_list_files, sd, 'val')


copy_files(src_dir)

print ("end")
