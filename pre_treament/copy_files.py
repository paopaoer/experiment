import os
import re
import shutil

src_dir = '/home/lss/ModelNet10/ModelNet10'
train_dst_dir = '/home/lss/PycharmProjects/cnn/test/1/train'
val_dst_dir = '/home/lss/PycharmProjects/cnn/test/1/val'


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


copy_files(src_dir)
print("end")
            
