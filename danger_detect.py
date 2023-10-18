from ultralytics import YOLO
from typing import List
import numpy as np
import time
import cv2
import math
import lanedetect as ld
import pandas as pd
import datetime

MOUNT_HEIGHT = 135.5 #カメラ視野高さ(cm)
FOV_H = 60/2 #カメラ左右広角 Dynabook RX73:50
FOV_V = 44/2
YSIZE=480 #縦サイズ(1536x2016; 832x1088;q 480x640)
XSIZE=640 #横サイズ

model = YOLO('yolov8n.pt') # best0709.pt yolov8n.pt
source = "0"
tracker='botsort.yaml'
cap = cv2.VideoCapture(0)
detects = pd.DataFrame(columns=["id", "newy", "newd", "newxan", "last_accessed"]) #前フレームの情報を格納するデータフレーム

def cleanup_old_entries():
    global detects
    now = datetime.datetime.now()
    thirty_seconds_ago = now - datetime.timedelta(seconds=5)
    # Only keep rows accessed within the last 30 seconds
    detects = detects[detects['last_accessed'] > thirty_seconds_ago]

def access_or_add(id, newy=None, newd=None, newxan=None):
    global detects
    now = datetime.datetime.now()
    # Check if id exists
    if id in detects['id'].values:
        detects.loc[detects['id'] == id, 'last_accessed'] = now
        detects.loc[detects['id'] == id, 'newy'] = newy
        detects.loc[detects['id'] == id, 'newd'] = newd
        detects.loc[detects['id'] == id, 'newxan'] = newxan
    else:
        # Add new row with the provided details
        detects = detects._append({"id": id, "newy": newy, "newd": newd, "newxan":newxan, "last_accessed": now}, ignore_index=True)

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

def panel_ctrl(lane, aprc, dist, spd):
    if aprc=="Passing":
        #red
    elif dist/100<=10 or (dist/100<30 and (aprc=="Approaching" or spd>0)):
        #red blink
    elif dist/100>30 and (aprc!="Far"):
        #yellow
    else:
        #green

while True:
    ret, frame= cap.read()
    if not ret: #入力なし
        break
    initialtime = time.perf_counter() #処理開始時間
    results = model.track(frame, imgsz=XSIZE, conf=0.3, iou=0.5, persist=True, show=False, tracker='botsort.yaml')#load track model and get boxes
    #車線探知##############
    copy = np.copy(frame)
    edges = ld.canny(copy)
    isolated = ld.region(edges) #マスク
    lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = ld.average(copy, lines) #車線あり：探知、車線なし：デフォルト（およそバイクの幅）；二つの線を引く点の座標を出力
    lx1, ly1, lx2, ly2 = averaged_lines[0][0], averaged_lines[0][1], averaged_lines[0][2], averaged_lines[0][3]
    rx1, ry1, rx2, ry2 = averaged_lines[1][0], averaged_lines[1][1], averaged_lines[1][2], averaged_lines[1][3]
    al, bl, cl = ly2-ly1, lx1-lx2, lx2*ly1-lx1*ly2 #車線近似直線方程式
    ar, br, cr = ry2-ry1, rx1-rx2, rx2*ry1-rx1*ry2
    #　↓　実験用車線表示
    black_lines = ld.display_lines(copy, averaged_lines)
    frame = cv2.addWeighted(copy, 0.8, black_lines, 1, 1)
    ###############################
    if results[0].boxes.id is None: #物体検出なし
        cv2.imshow("frame", frame)
        continue
    for box in results[0].boxes:
        coordinate = box.xyxy.cpu().numpy().astype(int)[0]
        xmid = int((coordinate[0]+coordinate[2])/2) #枠下線の中心点ｘ座標
        ydown = coordinate[3] #枠下線のｙ座標
        #左右自車線判断##############
        dl = al*xmid + bl*ydown + cl #画像に映った左レーン（リアカメラ左側⇒右レーン）
        dr = ar*xmid + br*ydown + cr #画像に映った右レーン
        if dl > 0 and dr > 0:
            lane_info = "right"
        elif (dl<= 0 and dr >=0) or (dl > 0 and dr < 0):
            lane_info = "same"
        else:
            lane_info = "left"
        ############################
        h = box.xywh.cpu().numpy().astype(int)[0][3]
        boxclass = box.cls.cpu().numpy().astype(int)
        id = box.id.cpu().numpy().astype(int)[0]
        i = 0
        lasty = 0 #物体毎にlastリセット
        lastdist = 0
        lastanglex = 0
        dist_an = distance_angle(xmid,ydown)
        dist, angle_y, angle_x = dist_an[0], dist_an[1], dist_an[2]

        #データフレームから前フレームの情報を取り出すかつデータフレームを更新/追加#########
        if id not in detects['id'].values: #存在しなかった目標
            lasty=0
            lastdist=-1
            lastanglex=0
        else: #存在する目標、過去のデータを取り出す
            lasty=detects.loc[detects['id'] == id, 'newy'].values[0]
            lastdist=detects.loc[detects['id'] == id, 'newd'].values[0]
            lastanglex=detects.loc[detects['id'] == id, 'newxan'].values[0]
        access_or_add(id, ydown, dist, angle_x) #更新または追加
        ##############################################################################

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

        #パネルコントロール################
        # if lane_info=="right":
        #     panel_ctrl(2, approach_info, dist, spd)
        # elif lane_info == "left":
        #     panel_ctrl(0, approach_info, dist, spd)
        # else:
        #     panel_ctrl(1, approach_info, dist, spd)

        #################################

        #　↓　実験用表示UI
        cv2.rectangle(frame, (coordinate[0], coordinate[1]), (coordinate[2], coordinate[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"Id.{id}, Dist.{dist/100:.2f}m", (coordinate[0], coordinate[1]),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness = 1, )
        cv2.putText(frame, f"Spd:{spd/100:.2f}m/s", (coordinate[0], coordinate[1]+20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness = 1, )
        cv2.putText(frame, f"{approach_info}", (coordinate[0], coordinate[1]+40),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness = 1, )
        cv2.drawMarker(frame, (xmid, ydown), (255, 0, 0), thickness = 2)
        cv2.putText(frame, f"{lane_info}", (xmid+10, ydown+10),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness = 2, )
        cv2.putText(frame, f"({xmid},{ydown})", (xmid, ydown),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness = 1, )
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
