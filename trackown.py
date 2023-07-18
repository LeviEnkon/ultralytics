import ultralytics
from ultralytics import YOLO
#supervision
import supervision
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator

from typing import List
import numpy as np
import time
import cv2

model = YOLO('yolov8n.pt')
#results=model.track(source='0', show=True, tracker='botsort.yaml')
source = "0"
racker='botsort.yaml'
cap = cv2.VideoCapture(0)

lastheight=0

detects = [] #0:id 1:lasth 2:h

def check_approach(lasth, h):
    if lasth != 0 and h>lasth:
        return True
    else:
        return False

while True:
    ret, frame= cap.read()
    if not ret: #入力なし
        break
    results = model.track(frame, conf=0.3, iou=0.5, persist=True, show=True)
    if results[0].boxes.id is None: #物体検出なし
        pass

    # boxes = results[0].boxes.xywh.cpu().numpy().astype(int) #物体座標
    # ids = results[0].boxes.id.cpu().numpy().astype(int) #物体番号
    # classes = results[0].boxes.cls.cpu().numpy().astype(int) #物体クラス
    # height = boxes[3] #物体高さ
    for box in results[0].boxes:
        coordinate = box.xywh.cpu().numpy().astype(int)
        h = coordinate[0][3]
        boxclass = box.cls.cpu().numpy().astype(int)
        id = box.id.cpu().numpy().astype(int)[0]
        i = 0

        lastheight = 0 #物体毎にlasthリセット
        while i <len(detects): #探知されたもののリストからidを探す
            if detects[i][0]==id:
                lastheight = detects[i][1] #以前格納された高さを取り出す
                detects[i][1] = detects[i][2]
                detects[i][2] = h #新しい高さをlasthに格納
                break
            i+=1
        if i==len(detects): #IDはリストに存在しない
            detects.append([id, 0, h])

        if check_approach(lastheight, h)==True:
            print("物体",id,"が接近中!")
        print("物体id", id, "物体種類", boxclass, "座標", coordinate[0][0], 
              ",", coordinate[0][1], "h=", h)
    print("#####################")
    time.sleep(1) #force 1 fps

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    # for box in result.boxes:
    #     print("class=", box[0].cls, "coord=", box[0].xywh)
    # print("one frame ###########")