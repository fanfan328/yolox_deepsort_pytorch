import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from loguru import logger
import sys
sys.path.append('../deep_sort')

from deep_sort.network.model import CNN

class GetFeatures(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = CNN(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
        self.net.load_state_dict(state_dict)

        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        # self.size = (64, 128)
        self.norm=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128,64)),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225))
        ])


    def __call__(self, im_crops):

        im_batch = im_batch = torch.cat([self.norm(im.astype(np.float32)/255.).unsqueeze(0) for im in im_crops], dim=0).float()
        # logger.info("Image has been - Preprocess")
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


if __name__ == '__main__':
    # img = cv2.imread("person_1.jpg")[:,:,(2,1,0)]
    img = cv2.imread("person_1.jpg")
    out = GetFeatures("checkpoint/ckpt.pth")
    feature = out(img)
    print(feature.shape)

