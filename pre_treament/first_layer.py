import numpy as np
from skimage import io,transform
import os

src_dir = '/home/lss/ModelNet10/ModelNet10'
train_dst_dir = '/home/lss/PycharmProjects/cnn/test/2/train'
val_dst_dir = '/home/lss/PycharmProjects/cnn/test/2/val'
sub_dirs=['bed','chair','dresser','monitor','sofa','table','toilet']




size=112


def create_dirs(src_dir):

    #sub_dirs = os.listdir(src_dir)
    print(sub_dirs)
    for sd in sub_dirs:
        if sd.find('.DS Store') > -1:
            continue
        if not os.path.exists(train_dst_dir+'/'+sd):
            os.mkdir(train_dst_dir+'/'+sd)
        if not os.path.exists(val_dst_dir+'/'+sd):
            os.mkdir(val_dst_dir+'/'+sd)
    return sub_dirs



def max_pool_2d(img):
    height,width,channel=img.shape

    new_image = np.zeros((int(height/2), int(width/2), channel), np.uint8)

    for k in range(0,channel,1):
        for i in range(0,height-2,2):
            for j in range(0,width-2,2):
                max_pixel=max(img[i][j][k],img[i+1][j][k],img[i][j+1][k],img[i+1][j+1][k])
                new_image[int(i/2)][int(j/2)][k]=max_pixel
            

    return new_image


def combinate_images(path, sd, phase):

    k=0
    list_dirs = os.listdir(path)
    list_dirs.sort()

    list_images=[]

    for tlf in list_dirs:
        if tlf.find('.DS_Store') > -1:
            continue
        if (tlf.find('.off')) > -1:
            continue
        img=io.imread(path+'/'+tlf)
        #img=max_pool_2d(img)
        list_images.append(img)

        k += 1

        if k == 14:
            #print(list_images[0].shape)
            list_images=list_images[5::]
            empty_image = np.zeros((size*3, size*3, 4), np.uint8)
            
            for i in range(3):
                for j in range(3):
                    for m in range(size):
                        for n in range(size):
                            empty_image[m+i*size][n+j*size]=list_images[i*3+j][m][n]
            
            
            
            index = tlf.rfind('_')
            image_name = tlf[:index]+'.png'
            if(phase == 'train'):
                dest_file = train_dst_dir+'/'+sd+'/'+image_name
            else:
                dest_file = val_dst_dir+'/'+sd+'/'+image_name
            print (dest_file)
            if not os.path.exists(dest_file):
                io.imsave(dest_file, empty_image)


            list_images=[]
            k=0

def copy_files(src_dir):
    #sub_dirs =create_dirs(src_dir)
    create_dirs(src_dir)
    for sd in sub_dirs:
        train_list_files = src_dir+'/'+sd+'/'+'train'
        combinate_images(train_list_files, sd, 'train')

        val_list_files = src_dir+'/'+sd+'/'+'test'
        combinate_images(val_list_files, sd, 'val')


copy_files(src_dir)

print ("end")










