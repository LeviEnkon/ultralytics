import cv2
import matplotlib.pyplot as plt
import numpy as np
import lanedetect as ld
from ultralytics import YOLO

image_path = r"./testcross.jpg"
image1 = cv2.imread(image_path)
image1 = cv2.resize(image1, dsize=(640,480))
copy = np.copy(image1)
edges = ld.canny(copy)
isolated = ld.region(edges) #マスク
lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_lines = ld.average(copy, lines)
lx1, ly1, lx2, ly2 = averaged_lines[0][0], averaged_lines[0][1], averaged_lines[0][2], averaged_lines[0][3]
rx1, ry1, rx2, ry2 = averaged_lines[1][0], averaged_lines[1][1], averaged_lines[1][2], averaged_lines[1][3]
al, bl, cl = ly2-ly1, lx1-lx2, lx2*ly1-lx1*ly2
ar, br, cr = ry2-ry1, rx1-rx2, rx2*ry1-rx1*ry2

lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_lines = ld.average(copy, lines)
print("line", averaged_lines)
black_lines = ld.display_lines(copy, averaged_lines)
lane = cv2.addWeighted(copy, 0.8, black_lines, 1, 1)

model = YOLO('best0709.pt')
results = model.track(image1, imgsz=640, conf=0.3, iou=0.5, persist=True, show=False, tracker='botsort.yaml')
for box in results[0].boxes:
    coordinate = box.xyxy.cpu().numpy().astype(int)[0]
    xmid = int((coordinate[0]+coordinate[2])/2) #枠下線の中心点ｘ座標
    ydown = coordinate[3]

    p = (xmid,ydown)
    dl = al*p[0] + bl*p[1] + cl
    dr = ar*p[0] + br*p[1] + cr
    if dl > 0 and dr > 0:
        info = "right"
        print("right")
    elif (dl<= 0 and dr >=0) or (dl > 0 and dr < 0):
        info = "same"
        print("middle")
    else:
        info = "left"
        print("left")
    cv2.rectangle(lane, (coordinate[0], coordinate[1]), (coordinate[2], coordinate[3]), (0, 255, 0), 2)
    cv2.drawMarker(lane, p, (255, 0, 0), thickness = 2)
    cv2.putText(lane, f"{info}", (p[0]+10, p[1]+10),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness = 2, )

cv2.imshow("lane",lane)
cv2.waitKey(0)
cv2.destroyAllWindows()