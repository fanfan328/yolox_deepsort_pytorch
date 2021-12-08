# from detection import Predictor
import imutils, argparse, cv2
from imutils.video import count_frames
import time
import loguru as logger
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys

sys.path.insert(0, './YOLOX')
from YOLOX.yolox.data.datasets.coco_classes import COCO_CLASSES
from detection import Predictor  # From Object Detection
from deep_sort.utils.parser import get_config
from deep_sort import DeepSort
import torch

from utils.visualize import vis_track
import numpy as np

# from PyQt5.QtWidgets import QApplication
# from videoplayer1 import VideoWindow
# from videoplayer1 import ProgressBar

class_names = COCO_CLASSES


class ObjectTracker(object):
    def __init__(self):
        super(ObjectTracker, self).__init__()

    def tes(self):
        print(f"Ini masuk function")

    def tracking_deepsort(self, file, filter_class='person'):
        # From objectdetection
        info_obj = self.detector.detect(file, flag_vis=False)
        outputs = []

        # Feed every prediction into the deepsort
        if info_obj['box_nums'] > 0:
            bbox_xywh = []
            scores = []

            for (x1, y1, x2, y2), class_id, score in zip(info_obj['boxes'], info_obj['cls_id'], info_obj['scores']):
                if filter_class and class_names[int(class_id)] not in filter_class:
                    continue
                bbox_xywh.append([int((x1 + x2) / 2), int((y1 + y2) / 2), x2 - x1, y2 - y1])
                scores.append(score)
            bbox_xywh = torch.Tensor(bbox_xywh)
            outputs = self.tracker.update(bbox_xywh, scores, file)
            # print(f" Boxes {len(info_obj['boxes'])} = {len(bbox_xywh)} Scores {len(info_obj['scores'])} = {len(scores)} Output {len(outputs)}")

            if (len(outputs) < len(scores)):
                image, self.person_id_array = vis_track(file, outputs, scores, self.person_id_array)
            else:
                image = file

        return image, outputs

    def track_video(self, args, num_classes, progressbar='X', model='yolox-s', ckpt='yolox_s.pth'):
        file = args.path
        model_type = args.model

        # Initiate Predictor and DeepSort
        self.detector = Predictor(model, ckpt)
        self.progress = 0.0
        cfg = get_config()
        cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        self.tracker = DeepSort(cfg.DEEPSORT.REID_CKPT, num_classes=num_classes, model=model_type,
                                max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                                nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT,
                                nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                use_cuda=True)

        # Initiate the input file (Breakdown from the video into single frame)
        cap = cv2.VideoCapture(file)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)

        # count the total number of frames in the video file
        total = count_frames(file, override=False)
        print("[INFO] {:,} total frames read from {}".format(total, file[file.rfind(os.path.sep) + 1:]))

        # Declare the output path and file
        save_folder = os.path.abspath(os.getcwd())
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, "OUT_" + file.rsplit('/', 1)[1])
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        vid_writer = cv2.VideoWriter(
            save_path, fourcc, fps, (int(width), int(height))
        )
        # print(f"Detail Video = {fps} - {width}-{height}")

        # Start Processing the video (Breakingdown into frames)
        counter = 0
        start_time = time.time()
        start_process_time = time.time()

        self.person_id_array = []
        while True:
            _, frame = cap.read()
            if frame is None:
                break

            # Process Tracker
            # frame = imutils.resize(frame, height=500)
            frame, _ = self.tracking_deepsort(frame)

            # cv2.imshow('show', frame)
            # cv2.imwrite(os.path.join(save_folder,'test/'+str(counter)+'.jpg'), frame)
            vid_writer.write(frame)
            counter += 1
            if counter % 10 == 0:
                self.progress = counter / total * 100
                print(
                    f"Progress = {counter}/{total} - {counter / total * 100:.2f}% - {time.time() - start_process_time:.2f}s")
                if (progressbar != 'X'):
                    progressbar.setVal(int(self.progress))
                start_process_time = time.time()
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break

        vid_writer.release()
        cap.release()

        cv2.destroyAllWindows()

        # for i in range(len(frame_final)):
        #   vid_writer.write(frame_final[i])

        print(f"Total Processing = {time.time() - start_time:.2f}s")
        return self.person_id_array, save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser("YOLOX-Tracker Demo!")
    parser.add_argument('-p', "--path", type=str, help="choose a video")
    parser.add_argument('-m', "--model", type=int, default=1, help="model used")
    parser.add_argument('-d', "--dataset", type=str, help="dataset")
    args = parser.parse_args()

    num_classes = 751
    if args.dataset == 'spacejam':
        num_classes = 32673
    elif args.dataset == 'pep':
        num_classes = 1264

    tracker = ObjectTracker()
    if os.path.isfile(args.path):
        print('Video Process')
        list_id_person, out_video = tracker.track_video(args, num_classes)
    else:
        print('Nothing Process')
