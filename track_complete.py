import ultralytics
from ultralytics import YOLO

from typing import List
import numpy as np
import time
import cv2

import math
import lanedetect as ld

MOUNT_HEIGHT = 135.5 #カメラ視野高さ(cm)
FOV_H = 60/2 #カメラ左右広角 Dynabook RX73:50
FOV_V = 44/2
YSIZE=480 #縦サイズ(1536x2016; 832x1088;q 480x640)
XSIZE=640 #横サイズ

model = YOLO('yolov8n.pt')
#results=model.track(source='0', show=True, tracker='botsort.yaml')
source = "0"
tracker='botsort.yaml'
cap = cv2.VideoCapture(0)

detects = [] #0:id 1:lastydown 2:newydown 3:lastdistance 4:newdistance

def check_approach(lasty, y):
    if (lasty != 0 and y>lasty) or y>=YSIZE-1:
        return True
    else:
        return False


def distance_angle(x,y):
    angle_x = abs(x-XSIZE/2)/(XSIZE/2) * FOV_H
    angle_y = (y-YSIZE/2)/(YSIZE/2) * FOV_V
    if y <= YSIZE/2:
        dist = -1
    elif y >= YSIZE-1:
        dist = 0
    else:
        dist_y = MOUNT_HEIGHT / math.tan(math.radians(angle_y))
        dist_x = dist_y * math.tan(math.radians(angle_x))
        dist=math.sqrt(dist_x*dist_x + dist_y*dist_y)
    return [dist, angle_y, angle_x]

def speed(lastdist, dist, lastanglex, anglex, t):
    if lastdist==-1:
        return -1
    return -(dist*math.cos(math.radians(anglex))-lastdist*math.cos(math.radians(lastanglex)))/t

while True:
    ret, frame= cap.read()
    if not ret: #入力なし
        break
    initialtime = time.perf_counter() #処理開始時間
    results = model.track(frame, imgsz=XSIZE, conf=0.3, iou=0.5, persist=True, show=False, tracker='botsort.yaml')
    inferencetime = time.perf_counter() - initialtime
    # print("時間差：", inferencetime, "s")

    if results[0].boxes.id is None: #物体検出なし
        cv2.imshow("frame", frame)
        continue

    # boxes = results[0].boxes.xywh.cpu().numpy().astype(int) #物体座標
    # ids = results[0].boxes.id.cpu().numpy().astype(int) #物体番号
    # classes = results[0].boxes.cls.cpu().numpy().astype(int) #物体クラス
    # height = boxes[3] #物体高さ
    for box in results[0].boxes:
        coordinate = box.xyxy.cpu().numpy().astype(int)[0]
        print("角座標",coordinate)
        xmid = int((coordinate[0]+coordinate[2])/2) #枠下線の中心点ｘ座標
        ydown = coordinate[3] #枠下線のｙ座標
        h = box.xywh.cpu().numpy().astype(int)[0][3]
        boxclass = box.cls.cpu().numpy().astype(int)
        id = box.id.cpu().numpy().astype(int)[0]
        
        i = 0
        lasty = 0 #物体毎にlasthリセット
        lastdist = 0
        lastanglex = 0

        dist_an = distance_angle(xmid,ydown)
        dist, angle_y, angle_x = dist_an[0], dist_an[1], dist_an[2]

        if dist == -1:
            print("物体",id,"が十分遠い")
        elif dist == 0:
            print("物体",id,"が探知範囲を経過")
        else:
            print("物体",id,"の距離は ",dist/100," m")

        while i <len(detects): #探知されたもののリストからidを探す
            if detects[i][0]==id:
                #前後フレームｙ座標
                detects[i][1] = detects[i][2] #前回の高さをlastyにする
                lasty = detects[i][1] #以前格納されたydownさを取り出す
                detects[i][2] = ydown #新しい高さをlastydownに格納
                #前後フレーム距離
                detects[i][3] = detects[i][4]
                lastdist = detects[i][3]
                detects[i][4] = dist
                #前後フレームx角度
                detects[i][5] = detects[i][6]
                lastanglex = detects[i][5]
                detects[i][6] = angle_x
                break
            i+=1
        if i==len(detects): #IDはリストに存在しない
            detects.append([id, 0, ydown, -1, dist, 0, angle_x])
        spd=speed(lastdist, dist, lastanglex, angle_x, inferencetime)

        if ydown>=YSIZE-1:
            info="Passing"
        elif ydown<240:
            info="Far"
        else:
            if check_approach(lasty, ydown)==True:
                info="Approaching"
            else:
                info="No Approach"
        

        print("物体id", id, "物体種類", boxclass, "中下座標", xmid, ",", ydown, "h=", h)
        cv2.rectangle(frame, (coordinate[0], coordinate[1]), (coordinate[2], coordinate[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"Id.{id}, Dist.{dist/100:.2f}m", (coordinate[0], coordinate[1]),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness = 1, )
        cv2.putText(frame, f"Spd:{spd/100:.2f}m/s", (coordinate[0], coordinate[1]+20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness = 1, )
        cv2.putText(frame, f"{info}", (coordinate[0], coordinate[1]+40),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness = 1, )
        cv2.drawMarker(frame, (xmid, ydown), (255, 0, 0), thickness = 2)
        cv2.putText(frame, f"({xmid},{ydown})", (xmid, ydown),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness = 1, )
    print("#####################")
    # time.sleep(2) #force 1 fps
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
