
import numpy as np
import pandas as pd
import cv2
import os


def resize_images(img_path,path_to_save):
    for files in sorted(os.listdir(img_path)):
        try:
            img=cv2.imread(files)
            #print(img.shape)
            size=(new_im_wd,new_im_ht)
            cv2.imwrite(os.path.join(path_to_save,files),cv2.resize(img,size))
        except:
            print(files)


def mod_annot(data,img_path,path_to_save,new_im_wd,new_im_ht):

    size=(new_im_wd,new_im_ht)

    for i in range(len(data)):
        file=data['frame'][i]
        img=cv2.imread(os.path.join(img_path,file))
        cv2.imwrite(os.path.join(path_to_save,file),cv2.resize(img,size))
        #data.frame[i]='mod_'+data.frame[i]
        data.xmin[i]=round(new_im_wd*data.xmin[i]/img.shape[1])
        data.ymin[i]=round(new_im_ht*data.ymin[i]/img.shape[0])
        data.xmax[i]=round(new_im_wd*data.xmax[i]/img.shape[1])
        data.ymax[i]=round(new_im_ht*data.ymax[i]/img.shape[0])
    return data


def mod_annotations(img_path,path_to_save,new_im_wd,new_im_ht):
    #img_path='/home/lenovo/object-detection/Keras_object-detection/pretrained_MobileNet_keras/weapon_dataset/images'
    #path_to_save='/home/lenovo/object-detection/Keras_object-detection/pretrained_MobileNet_keras/weapon_dataset/images_mod'
    cols=['frame','xmin','xmax','ymin','ymax','class_id']
    train_annot=pd.read_csv(os.path.join(img_path,'train_labels.csv'))[cols]
    test_annot=pd.read_csv(os.path.join(img_path,'test_labels.csv'))[cols]
    #train_annot.head()


    data_train=mod_annot(train_annot,img_path,path_to_save,new_im_wd,new_im_ht)
    data_train.to_csv(os.path.join(path_to_save,'train_labels.csv'),index=False)

    data_test=mod_annot(test_annot,img_path,path_to_save,new_im_wd,new_im_ht)
    data_test.to_csv(os.path.join(path_to_save,'test_labels.csv'),index=False)
