import cv2
import matplotlib.pyplot as plt
import numpy as np
import lanedetect as ld

image_path = r"./testwhiteline.jpg"
image1 = cv2.imread(image_path)
copy = np.copy(image1)
edges = ld.canny(copy)
isolated = ld.region(edges)
lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_lines = ld.average(copy, lines)

p = (1432,693)
lx1, ly1, lx2, ly2 = averaged_lines[0][0], averaged_lines[0][1], averaged_lines[0][2], averaged_lines[0][3]
rx1, ry1, rx2, ry2 = averaged_lines[1][0], averaged_lines[1][1], averaged_lines[1][2], averaged_lines[1][3]

al, bl, cl = ly2-ly1, lx1-lx2, lx2*ly1-lx1*ly2
ar, br, cr = ry2-ry1, rx1-rx2, rx2*ry1-rx1*ry2
dl = al*p[0] + bl*p[1] + cl
dr = ar*p[0] + br*p[1] + cr
if dl > 0 :
    print("left")
elif dl<= 0 and dr >=0:
    print("middle")
else:
    print("right")