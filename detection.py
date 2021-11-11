import cv2
import numpy as np
import torch
import time
import os
from loguru import logger

# import sys
# sys.path.insert(0, './detector/YOLOX')
from YOLOX.yolox.data.data_augment import ValTransform
from YOLOX.yolox.data.datasets import COCO_CLASSES
from YOLOX.yolox.exp.build import get_exp_by_name
from YOLOX.yolox.utils import postprocess
from utils.visualize import vis

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

class Predictor(object):
    def __init__(
        self,
        model='yolox-s', 
        ckpt='yolox_s.pth',
        cls_names=COCO_CLASSES,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model
        self.exp = get_exp_by_name(model)
        self.cls_names = cls_names
        self.num_classes = self.exp.num_classes
        self.confthre = self.exp.test_conf
        self.nmsthre = self.exp.nmsthre
        self.test_size = self.exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)

        self.model = self.exp.get_model()
        self.model.to(self.device)
        self.model.eval()
        checkpoint = torch.load(ckpt, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])


    def detect(self, img, flag_vis=True, cls_conf=0.35): #can be change into another score
        img_info = {}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        img_info["raw_img"] = img
        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        img = img.to(self.device)

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )[0].cpu().numpy()
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        
        # After Detection, Send it into visual function
        bboxes = outputs[:, 0:4]

        # print(outputs)
        img_info['boxes'] = outputs[:, 0:4]/ratio
        img_info['scores'] = outputs[:, 4] * outputs[:, 5]
        img_info['cls_id'] = outputs[:, 6]
        img_info['box_nums'] = outputs.shape[0]

        # print(f"Boxes = {info['boxes']} Scores = {info['scores']} Conf = {conf}")
        if flag_vis:
            img_info['visual'] = vis(img_info['raw_img'], img_info['boxes'], img_info['scores'], img_info['cls_id'], cls_conf, COCO_CLASSES)
        return img_info
        # return outputs, img_info

if __name__=='__main__':
    detector = Predictor()
    img = cv2.imread('YOLOX/assets/dog.jpg')
    out = detector.detect(img)
    cv2.imshow('demo', out['visual'])
    cv2.waitKey(0)
    print(out)