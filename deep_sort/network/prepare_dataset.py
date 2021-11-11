"""
This file to process raw data from the dataset, so it can be processed into ImageFolder (torchvision)
https://pytorch.org/vision/stable/datasets.html

"""
import os
import cv2
from shutil import copyfile

download_path = 'dataset'

if not os.path.isdir(download_path):
    print('Dataset Folder is not Found')

save_path = download_path + '/data_prepared'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

train_save_path = save_path + '/train'
val_save_path = save_path + '/val'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
    os.mkdir(val_save_path)

counter_total_data=0
for root, dirs, files in os.walk(download_path, topdown=True):
    for name in files:
        if not name[-3:]=='mp4':
            continue
        ID  = name[:-4].split('_')
        print(f"{name} - {ID}")
        cap = cv2.VideoCapture('dataset/'+name)
        tr_path = train_save_path + '/' + ID[0]
        if not os.path.isdir(tr_path):
            os.mkdir(tr_path)
            val_path = val_save_path + '/' + ID[0]  #first image is used as val image
            os.mkdir(val_path)

        counter_data=0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if frame is None:
                break
            else:
                if counter_data==1:
                    cv2.imwrite(val_path+'/'+ID[0]+'_'+str(counter_data)+'.jpg',frame)
                else:
                    cv2.imwrite(tr_path+'/'+ID[0]+'_'+str(counter_data)+'.jpg',frame)
                counter_data+=1
                counter_total_data+=1
        
        cap.release()
        cv2.destroyAllWindows()
        # src_path = train_path + '/' + name

        # copyfile(src_path, dst_path + '/' + name)