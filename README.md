# yolox_deepsort_pytorch
This project is to do object detection in this case human and do the tracking. Also we made the GUI for this project using PyQT

Installation
1. Clone the repository :
   ```bash
   git clone --recurse-submodules https://github.com/fanfan328/yolox_deepsort_pytorch.git
   ```
   
  * Don't forget to use `--recurse-submodules` to pull the YoloX also
  * If you forget to use the command above, you can run this command `git submodule update --init`
  
 2. Make sure that you fulfill all the requirements: 
   ```bash
   pip install -r requirements.txt
   ```
   
3. Download the pre-trained weight for the YoloX, For this case using YoloX-s but you can modify depends on your requirement
    (Source https://github.com/Megvii-BaseDetection/YOLOX) Put on your root directory project
    
4. Also for weight of deepSort, you can either train on yourself, or use the existing one on checkpoint folder.
  (Don't forget to set-up the checkpoint on deep_sort/configs/deep_sort.yaml)
  
5. To run the test, you can choose either using GUI mode
    ```bash
   python videoplayer1.py
   ```
   
    Or directly using command on object_tracker
    
    ```bash
   python object_tracker.py --path=<video_path.mp4> --model=2 
   ```
    
