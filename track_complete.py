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
source = "0"
tracker='botsort.yaml'
cap = cv2.VideoCapture(0)

detects = [] #0:id 1:last ydown 2:new ydown 3:last distance 4:new distance 5:last x angle 6: new x angle

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
    #load track model and get boxes
    results = model.track(frame, imgsz=XSIZE, conf=0.3, iou=0.5, persist=True, show=False, tracker='botsort.yaml')
    #lane detect
    copy = np.copy(frame)
    edges = ld.canny(copy)
    isolated = ld.region(edges) #マスク
    lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = ld.average(copy, lines) #車線あり：探知、車線なし：デフォルト（およそバイクの幅）；二つの線を引く点の座標を出力
    lx1, ly1, lx2, ly2 = averaged_lines[0][0], averaged_lines[0][1], averaged_lines[0][2], averaged_lines[0][3]
    rx1, ry1, rx2, ry2 = averaged_lines[1][0], averaged_lines[1][1], averaged_lines[1][2], averaged_lines[1][3]
    al, bl, cl = ly2-ly1, lx1-lx2, lx2*ly1-lx1*ly2
    ar, br, cr = ry2-ry1, rx1-rx2, rx2*ry1-rx1*ry2

    if results[0].boxes.id is None: #物体検出なし
        cv2.imshow("frame", frame)
        continue
    for box in results[0].boxes:
        coordinate = box.xyxy.cpu().numpy().astype(int)[0]
        xmid = int((coordinate[0]+coordinate[2])/2) #枠下線の中心点ｘ座標
        ydown = coordinate[3] #枠下線のｙ座標
        #車線探知##############
        dl = al*xmid + bl*ydown + cl #画像に映った左のレーン（リアカメラ⇒右レーン）
        dr = ar*xmid + br*ydown + cr #画像に移った右のレーン
        if dl > 0 :
            lane_info = "right"
        elif dl<= 0 and dr >=0:
            lane_info = "same"
        else:
            lane_info = "left"
        #######################
       

        h = box.xywh.cpu().numpy().astype(int)[0][3]
        boxclass = box.cls.cpu().numpy().astype(int)
        id = box.id.cpu().numpy().astype(int)[0]
        
        i = 0
        lasty = 0 #物体毎にlasthリセット
        lastdist = 0
        lastanglex = 0

        dist_an = distance_angle(xmid,ydown)
        dist, angle_y, angle_x = dist_an[0], dist_an[1], dist_an[2]

        # if dist == -1:
        #     dist_info = "far"
        # elif dist == 0:
        #     dist_info = "passing"
        # else:
        #     print("物体",id,"の距離は ",dist/100," m")

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
                #前後フレーム水平ズレ角度
                detects[i][5] = detects[i][6]
                lastanglex = detects[i][5]
                detects[i][6] = angle_x
                break
            i+=1 #id がヒットしない
        if i==len(detects): #IDはリストに存在しない
            detects.append([id, 0, ydown, -1, dist, 0, angle_x])

        #接近探知　１フレーム下座標範囲外：通過　２フレーム下座標は画面の中線以上：遠い　３以外：前後フレームの下y座標で接近判断
        if ydown>=YSIZE-1:
            approach_info="Passing"
        elif ydown<YSIZE//2:
            approach_info="Far"
        else:
            if check_approach(lasty, ydown)==True:
                approach_info="Approaching"
            else:
                approach_info="No Approach"

        inferencetime = time.perf_counter() - initialtime #速度測定直前までの処理時間
        spd=speed(lastdist, dist, lastanglex, angle_x, inferencetime) #前後フレームの距離、処理時間と水平ズレ角⇒速度



        print("物体id", id, "物体種類", boxclass, "中下座標", xmid, ",", ydown, "h=", h)
        cv2.rectangle(frame, (coordinate[0], coordinate[1]), (coordinate[2], coordinate[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"Id.{id}, Dist.{dist/100:.2f}m", (coordinate[0], coordinate[1]),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness = 1, )
        cv2.putText(frame, f"Spd:{spd/100:.2f}m/s", (coordinate[0], coordinate[1]+20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness = 1, )
        cv2.putText(frame, f"{info}", (coordinate[0], coordinate[1]+40),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness = 1, )
        cv2.drawMarker(frame, (xmid, ydown), (255, 0, 0), thickness = 2)
        cv2.putText(frame, f"({xmid},{ydown})", (xmid, ydown),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness = 1, )
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
