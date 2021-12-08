"""
This file to process raw data from the dataset, so it can be processed into ImageFolder (torchvision)
https://pytorch.org/vision/stable/datasets.html

"""
import os
import shutil
import os.path as osp
import cv2
from shutil import copyfile

def train_test_split():
    download_path = '/home/limitrix/dev/ext1/_Class/advanced_programming_projects/data/dataset/dataset/examples'
    # download_path = 'dataset'

    if not os.path.isdir(download_path):
        print('Dataset Folder is not Found')

    save_path = '/home/limitrix/dev/ext1/_Class/advanced_programming_projects/data/dataset/dataset/processed'
    # save_path = download_path + '/data_prepared'
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
            cap = cv2.VideoCapture(os.path.join(download_path, name))

            tr_path = train_save_path + '/' + ID[0]
            if not os.path.isdir(tr_path):
                os.mkdir(tr_path)
            val_path = val_save_path + '/' + ID[0]  #first image is used as val image
            if not os.path.isdir(val_path):
                os.mkdir(val_path)

            counter_data=0
            while (cap.isOpened()):
                ret, frame = cap.read()
                if frame is None:
                    break
                else:
                    if counter_data==1:
                        cv2.imwrite(os.path.join(val_path, ID[0]+'_'+str(counter_data)+'.jpg'),frame)
                    else:
                        cv2.imwrite(os.path.join(tr_path, ID[0]+'_'+str(counter_data)+'.jpg'),frame)
                    counter_data+=1
                    counter_total_data+=1

            cap.release()
            cv2.destroyAllWindows()
            # src_path = train_path + '/' + name

            # copyfile(src_path, dst_path + '/' + name)

def pep_combine():
    sources = {
        "/home/limitrix/dev/ext1/_Class/advanced_programming_projects/data/pep_256x128/raw/scen1",
        "/home/limitrix/dev/ext1/_Class/advanced_programming_projects/data/pep_256x128/raw/scen2",
        "/home/limitrix/dev/ext1/_Class/advanced_programming_projects/data/pep_256x128/raw/scen3",
    }
    target = "/home/limitrix/dev/ext1/_Class/advanced_programming_projects/data/pep_256x128/combined"

    for source in sources:
        for i in os.listdir(source):
            com1 = os.path.join(source, i)
            for id in os.listdir(com1):
                src = osp.join(com1, id)
                tgt = osp.join(target, source.rsplit('/', 1)[1], id)
                if not osp.isdir(tgt):
                    os.makedirs(tgt)
                for img in os.listdir(src):
                    if not osp.isfile(osp.join(tgt, img)):
                        shutil.copy(osp.join(src, img), tgt)
                print(f"{src.split('/', 8)[-1]} -> {tgt.split('/', 8)[-1]}")

def pep_combine2():
    source = {
        "/home/limitrix/dev/ext1/_Class/advanced_programming_projects/data/pep_256x128/combined/scen1",
        "/home/limitrix/dev/ext1/_Class/advanced_programming_projects/data/pep_256x128/combined/scen2",
        "/home/limitrix/dev/ext1/_Class/advanced_programming_projects/data/pep_256x128/combined/scen3",
    }
    target = "/home/limitrix/dev/ext1/_Class/advanced_programming_projects/data/pep_256x128/all"
    test = 0
    train = 0
    for scen in source:
        for id in os.listdir(scen):
            src = osp.join(scen, id)
            if scen.rsplit('/', 1)[1] == 'scen1':
                test += 1
                tgt = osp.join(target, 'test', str(test))
            else:
                train += 1
                tgt = osp.join(target, 'train', str(train))
            if not osp.isdir(tgt):
                os.makedirs(tgt)
            for im in os.listdir(src):
                shutil.copy(osp.join(src, im), tgt)
            print(f"Copied {src.split('8',1)[1]} to {tgt.split('8',1)[1]}")

def train_val_split():
    train_src = "/home/limitrix/dev/ext1/_Class/advanced_programming_projects/data/pep_256x128/ready/train"
    val_src = train_src.rsplit('/',1)[0] + "/val"

    ids = os.listdir(train_src)
    train_len = int(len(ids) * 0.8)
    val = ids[train_len:]

    for i in val:
        shutil.move(osp.join(train_src, i), val_src)

if __name__ == "__main__":

    train_val_split()