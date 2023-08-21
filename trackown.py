import ultralytics
from ultralytics import YOLO

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
    initialtime = time.perf_counter() #処理開始時間
    results = model.track(frame, conf=0.3, iou=0.5, persist=True, show=False)
    inferencetime = time.perf_counter() - initialtime
    print("時間差：", inferencetime, "s")
    if results[0].boxes.id is None: #物体検出なし
        pass

    # boxes = results[0].boxes.xywh.cpu().numpy().astype(int) #物体座標
    # ids = results[0].boxes.id.cpu().numpy().astype(int) #物体番号
    # classes = results[0].boxes.cls.cpu().numpy().astype(int) #物体クラス
    # height = boxes[3] #物体高さ
    for box in results[0].boxes:
        coordinate = box.xyxy.cpu().numpy().astype(int)[0]
        print("角座標",coordinate)
        xmid = int((coordinate[0]+coordinate[2])/2)
        ydown = coordinate[3]
        h = box.xywh.cpu().numpy().astype(int)[0][3]
        boxclass = box.cls.cpu().numpy().astype(int)
        id = box.id.cpu().numpy().astype(int)[0]
        
        i = 0
        lastheight = 0 #物体毎にlasthリセット
        while i <len(detects): #探知されたもののリストからidを探す
            if detects[i][0]==id:
                detects[i][1] = detects[i][2] #前回の高さをlasthにする
                lastheight = detects[i][1] #以前格納された高さを取り出す
                detects[i][2] = h #新しい高さをlasthに格納
                break
            i+=1
        if i==len(detects): #IDはリストに存在しない
            detects.append([id, 0, h])

        if check_approach(lastheight, h)==True:
            print("物体",id,"が接近中!")
        else:
            print("接近なし")
        print("物体id", id, "物体種類", boxclass, "中下座標", xmid, ",", ydown, "h=", h)
        cv2.rectangle(frame, (coordinate[0], coordinate[1]), (coordinate[2], coordinate[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"Id {id}", (coordinate[0], coordinate[1]),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness = 2, )
        cv2.drawMarker(frame, (xmid, ydown), (255, 0, 0), thickness = 2)
    print("#####################")
    time.sleep(1) #force 1 fps
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
