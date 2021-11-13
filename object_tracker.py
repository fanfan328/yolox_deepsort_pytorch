# from detection import Predictor
import imutils, argparse, cv2
from imutils.video import count_frames
import time
import loguru as logger
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
sys.path.insert(0, './YOLOX')
from YOLOX.yolox.data.datasets.coco_classes import COCO_CLASSES
from detection import Predictor # From Object Detection
from deep_sort.utils.parser import get_config
from deep_sort import DeepSort
import torch

from utils.visualize import vis_track
import numpy as np


class_names = COCO_CLASSES

def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.
    Parameters
    """
    # print(f"Extract image with {bbox} - {patch_shape}")
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    # cv2.imshow('test', image)
    # cv2.waitKey(0)
    # print(tuple(patch_shape[::-1]))]
    return image

def find_color(roi, threshold=0.0):
  """
  Return the ratio of non white pixels to All pixels, white jersies should have low values
  """

  roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
  # set a min and max for team colors
  COLOR_MIN = np.array([0, 0, 0])
  COLOR_MAX = np.array([255, 255, 100])

  # dark teams will remain with this mask
  mask = cv2.inRange(roi_hsv, COLOR_MIN, COLOR_MAX)
  res = cv2.bitwise_and(roi,roi, mask= mask)

  # dark teams should have a higher ratio
  tot_pix = roi.any(axis=-1).sum()
  color_pix = res.any(axis=-1).sum()
  ratio = color_pix/tot_pix

  return(ratio)

def tracking_deepsort(file, detector, deepsort, filter_class='person'):
  # From objectdetection
  info_obj = detector.detect(file, flag_vis=False)
  outputs = []

  boxes = info_obj['boxes']
  # patches = [extract_image_patch(file, box, [box[3], box[2]]) for box in boxes]
  # color_ratios = [find_color(patch) for patch in patches]
  # save_folder = os.path.abspath(os.getcwd())
  # i=0
  # for patch in patches:
  #     i+=1
  #     cv2.imwrite(os.path.join(save_folder,'test/'+str(color_ratios[i]*100)+'.jpg'), patch)

  # Feed every prediction into the deepsort
  if info_obj['box_nums']>0:
      bbox_xywh = []
      scores = []
      #bbox_xywh = torch.zeros((info['box_nums'], 4))
      for (x1, y1, x2, y2), class_id, score  in zip(info_obj['boxes'],info_obj['cls_id'],info_obj['scores']):
          if filter_class and class_names[int(class_id)] not in filter_class:
              continue
          bbox_xywh.append([int((x1+x2)/2), int((y1+y2)/2), x2-x1, y2-y1])
          scores.append(score)
      bbox_xywh = torch.Tensor(bbox_xywh)
      outputs = deepsort.update(bbox_xywh, scores, file)
      image = vis_track(file, outputs, scores)

  return image, outputs
  
def track_video(file, model='yolox-s', ckpt='yolox_s.pth', filter_class=None):
  #Initiate Predictor and DeepSort
  detector = Predictor(model, ckpt)
  cfg = get_config()
  cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
  tracker = DeepSort(cfg.DEEPSORT.REID_CKPT,
                      max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                      nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                      max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                      use_cuda=True)
                      

  # Initiate the input file (Breakdown from the video into single frame)
  cap = cv2.VideoCapture(file)
  width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
  height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
  fps = cap.get(cv2.CAP_PROP_FPS)
  
  # count the total number of frames in the video file
  total = count_frames(file, override=False)
  print("[INFO] {:,} total frames read from {}".format(total,  file[file.rfind(os.path.sep) + 1:]))

  # Declare the output path and file
  save_folder = os.path.abspath(os.getcwd())
  os.makedirs(save_folder, exist_ok=True)
  save_path = os.path.join(save_folder, "OUT_"+file)
  fourcc = cv2.VideoWriter_fourcc(*'MP4V')
  vid_writer = cv2.VideoWriter(
    save_path, fourcc, fps, (int(width), int(height))
  )
  # print(f"Detail Video = {fps} - {width}-{height}")

  # Start Processing the video (Breakingdown into frames)
  counter = 0
  start_time=time.time()
  while True:
      _, frame = cap.read()
      if frame is None:
          break

      #Process Tracker
      start_process_time = time.time()
      # frame = imutils.resize(frame, height=500)
      frame,_ = tracking_deepsort(frame, detector, tracker)

      cv2.imshow('show', frame)
      cv2.imwrite(os.path.join(save_folder,'test/'+str(counter)+'.jpg'), frame) 
      vid_writer.write(frame) #Gak bisa write ?? 
      counter+=1
      if counter%9==0 :
        print(f"Progress = {counter}/{total} - {counter/total*100:.2f}% - {time.time() - start_process_time:.2f}s")
      ch = cv2.waitKey(1)
      if ch == 27 or ch == ord("q") or ch == ord("Q"):
          break

  vid_writer.release()
  cap.release()

  cv2.destroyAllWindows()

  # for i in range(len(frame_final)):
  #   vid_writer.write(frame_final[i])

  print(f"Total Processing = {time.time() - start_time:.2f}s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("YOLOX-Tracker Demo!")
    parser.add_argument('-p', "--path", type=str, help="choose a video")
    args = parser.parse_args()

    if os.path.isfile(args.path):
      print('Video Process')
      track_video(args.path)
    else:
      print('Nothing Process')