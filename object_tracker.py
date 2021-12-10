# from detection import Predictor
import imutils, argparse, cv2
from imutils.video import count_frames
import time
import skvideo
skvideo.setFFmpegPath('C:/ffmpeg/bin')
import skvideo.io
# print("FFmpeg version: {}".format(skvideo.getFFmpegVersion()))

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
        self.id_boxes = {}

    def tes(self):
        print(f"Ini masuk function")

    def memo_bbox(self, boxes):
        for i in range(len(boxes)):
            box = boxes[i]

            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            id = int(box[4])

            if id in self.id_boxes.keys():
                self.id_boxes[id].append([self.counter, x0, y0, x1, y1])
            else:
                self.id_boxes[id] = [[self.counter, x0, y0, x1, y1]]

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
        self.memo_bbox(outputs)

        return image, outputs

    def track_video(self, path, model_num=2, num_classes=751, progressbar='X', model='yolox-s', ckpt='yolox_s.pth'):
        # self.v_path = args.path
        # file = args.path
        # model_type = args.model
        self.v_path = path
        file = path
        model_type = model_num

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
        # save_path = os.path.join(save_folder, "OUT_" + file.rsplit('/', 1)[1])
        save_path = os.path.join(save_folder, "OUT_" + file)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        vid_writer = cv2.VideoWriter(
            save_path, fourcc, fps, (int(width), int(height))
        )
        # print(f"Detail Video = {fps} - {width}-{height}")

        # Start Processing the video (Breakingdown into frames)
        self.counter = 0
        start_time = time.time()
        start_process_time = time.time()

        self.person_id_array = []
        while True:
            _, frame = cap.read()
            if frame is None:
                break

            # Process Tracker
            # frame = imutils.resize(frame, height=500)
            self.counter += 1
            frame, _ = self.tracking_deepsort(frame)

            # cv2.imshow('show', frame)
            # cv2.imwrite(os.path.join(save_folder,'test/'+str(counter)+'.jpg'), frame)
            vid_writer.write(frame)
            if self.counter % 10 == 0:
                self.progress = self.counter / total * 100
                print(
                    f"Progress = {self.counter}/{total} - {self.counter / total * 100:.2f}% - {time.time() - start_process_time:.2f}s")
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

    def add_center(self, boxes):
        # centerized_boxes = []
        x_max = y_max = 0
        for box in boxes:
            x_dis = (box[3] - box[1]) / 2
            y_dis = (box[4] - box[2]) / 2
            x_center = int(x_dis + box[1])
            y_center = int(y_dis + box[2])
            x_max = x_dis if x_dis > x_max else x_max
            y_max = y_dis if y_dis > y_max else y_max
            box.extend([x_center, y_center])
        return boxes, x_max, y_max

    def fix_boxes(self, box, width, height):
        if box[1] < 0:
            box[3] = box[3] - box[1]
            box[1] = 0
        elif box[3] >= width:
            box[1] -= box[3] - width
            box[3] = width

        if box[2] < 0:
            box[4] = box[4] - box[2]
            box[2] = 0
        elif box[4] >= height:
            box[2] -= box[4] - height
            box[4] = height
        return box

    def crop_vid(self, id):
        boxes = self.id_boxes[id]
        boxes, x_max, y_max = self.add_center(boxes)

        y_rad = int(y_max * 1.5)
        x_rad = int(y_rad * 3 / 4)

        for box in boxes:
            box[1] = box[5] - x_rad
            box[3] = box[5] + x_rad
            box[2] = box[6] - y_rad
            box[4] = box[6] + y_rad

        # prepare frames
        # Initiate the input file (Breakdown from the video into single frame)
        cap = cv2.VideoCapture(self.v_path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

        # count the total number of frames in the video file
        total = count_frames(self.v_path, override=False)
        print("[INFO] {:,} total frames read from {}".format(total, self.v_path[self.v_path.rfind(os.path.sep) + 1:]))

        # Declare the output path and file
        save_folder = os.path.abspath(os.getcwd())
        os.makedirs(save_folder, exist_ok=True)
        # save_path = os.path.join(save_folder,"OUT_cropped_" + str(id) + "_" + self.v_path.rsplit('/', 1)[1].rsplit('.', 1)[0] + '.mp4')
        save_path = os.path.join(save_folder,"OUT_cropped_" + str(id) + "_" + self.v_path + '.mp4')
        out_video = np.empty([len(boxes), x_rad * 2, y_rad * 2, 3], dtype=np.uint8)
        out_video = out_video.astype(np.uint8)

        i = 0
        self.counter = 0
        self.progress = 0.0
        start_process_time = time.time()
        while True:
            _, frame = cap.read()
            if frame is None:
                break

            self.counter += 1
            if i == len(boxes):
                # vid_writer.release()
                skvideo.io.vwrite(save_path, out_video)
                cap.release()
                cv2.destroyAllWindows()
                continue
            if boxes[i][0] == self.counter:
                box = self.fix_boxes(boxes[i], width, height)
                n_frame = frame[int(box[1]):int(box[3]), int(box[2]):int(box[4])]
                # n_frame = cv2.resize(n_frame, [1920, 1080], interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(f'./img/nframe_{i}.jpg', n_frame)
                out_video[i] = cv2.cvtColor(n_frame, cv2.COLOR_BGR2RGB)
                i += 1
            elif boxes[i][0] < self.counter:
                raise Exception('Why is i is smaller than counter?')

            if self.counter % 10 == 0:
                self.progress = self.counter / total * 100
                print(
                    f"Progress = {self.counter}/{total} - {self.counter / total * 100:.2f}% - {time.time() - start_process_time:.2f}s")
                start_process_time = time.time()

            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        
        return save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser("YOLOX-Tracker Demo!")
    parser.add_argument('-p', "--path", type=str, help="choose a video")
    parser.add_argument('-m', "--model", type=int, default=1, help="model used")
    parser.add_argument('-d', "--dataset", type=str, help="dataset", choices=['spacejam', 'pep', 'viprior'])
    args = parser.parse_args()

    num_classes = 751
    if args.dataset == 'spacejam':
        num_classes = 32673
    elif args.dataset == 'pep':
        num_classes = 1264
    elif args.dataset == 'viprior':
        num_classes = 436

    tracker = ObjectTracker()
    if os.path.isfile(args.path):
        print('Video Process')
        list_id_person, out_video = tracker.track_video(args.path, args.model, num_classes)
        tracker.crop_vid(1)
    else:
        print('Nothing Process')
