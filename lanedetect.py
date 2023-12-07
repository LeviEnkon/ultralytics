import cv2
# import matplotlib.pyplot as plt
import numpy as np

#ref: https://medium.com/analytics-vidhya/building-a-lane-detection-system-f7a727c6694
# img = cv2.imread("./testwhitelne.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (5,5), 0)
# edges = cv2.Canny(img, 100, 200)
# img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
# lower_yellow = np.array([20, 100, 100], dtype = "uint8")
# upper_yellow = np.array([30, 255, 255], dtype = "uint8")
# mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
# mask_white = cv2.inRange(gray, 200, 255)
# mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
# mask_yw_image = cv2.bitwise_and(gray, mask_yw)

#convert into grey scale image
def grey(image):
  image=np.asarray(image)
  return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
#Gaussian blur to reduce noise and smoothen the image
def gauss(image):
  return cv2.GaussianBlur(image,(5,5),0)
#Canny edge detection
def canny(image):
    edges = cv2.Canny(image,200, 300)
    return edges

def region(image): #make mask, use the result of canny
    height, width = image.shape
    triangle = np.array([[(0, height), (width//2, height*1//2), (width, height)]])
    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask

def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    height, width = image.shape[:2]
    #make sure array isn't empty
    try:
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line
                #draw lines on a black image
                cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    except cv2.error:
        cv2.line(lines_image,(width//4, height), (width//2-35, 288), (255, 0, 0), 10)
        cv2.line(lines_image,(width*3//4, height), (width//2+35, 288), (255, 0, 0), 10)
    return lines_image

def make_points(image, average):
    # print(average)
    slope, y_int = average
    y1 = image.shape[0]
    #how long we want our lines to be --> 3/5 the size of the image
    y2 = int(y1 * (3/5))
    #determine algebraically
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return np.array([x1, y1, x2, y2])

def average(image, lines):
    left = []
    right = []
    height, width = image.shape[:2]
    try: #only process when no error
        if lines is not None:
            for line in lines:
                # print(line)
                x1, y1, x2, y2 = line.reshape(4)
                #fit line to points, return slope and y-int
                parameters = np.polyfit((x1, x2), (y1, y2), 1)
                # print(parameters)
                slope = parameters[0]
                y_int = parameters[1]
                #lines on the right have positive slope, and lines on the left have neg slope
                if slope < 0:
                    left.append((slope, y_int))
                else:
                    right.append((slope, y_int))
                    
            #takes average among all the columns (column0: slope, column1: y_int)
        right_avg = np.average(right, axis=0)
        left_avg = np.average(left, axis=0)
        #create lines based on averages calculates
        left_line = make_points(image, left_avg)
        right_line = make_points(image, right_avg)
    except TypeError:
        left_line = np.array([width//4, height, width//2-35, 288])
        right_line = np.array([width*3//4, height, width//2+35, 288])
    return np.array([left_line, right_line])


# image_path = r"./testwhiteline.jpg"
# image1 = cv2.imread(image_path)
# copy = np.copy(image1)
# edges = canny(copy)
# isolated = region(edges)
# plt.imshow(edges)
# plt.show()
# plt.imshow(isolated)
# plt.show()

# lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# averaged_lines = average(copy, lines)
# print("line", averaged_lines)
# black_lines = display_lines(copy, averaged_lines)
# plt.imshow(black_lines)
# plt.show()
# lanes = cv2.addWeighted(copy, 0.8, black_lines, 1, 1)
# plt.imshow(lanes)
# plt.show()