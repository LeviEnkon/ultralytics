import cv2
import numpy as np
from matplotlib import pyplot as plt

import matplotlib.image as mpimg
import os

def region(image, imshape):
  vertices = np.array([[(0,imshape[0]),(900, 600), (1000, 600), (imshape[1],imshape[0])]], dtype=np.int32)
  mask = np.zeros_like(image)
  mask = cv2.fillPoly(mask, vertices, 255)
  mask = cv2.bitwise_and(image, mask)
  return mask

def ldict(img):
    imshape = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(img, 200, 820)

    img_hsv=cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_yellow=np.array([20, 100, 100], dtype="uint8")
    upper_yellow=np.array([30, 255, 255], dtype="uint8")
    mask_yellow=cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_white=cv2.inRange(gray, 200, 255)
    mask_yw=cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image=cv2.bitwise_and(gray, mask_yw)

    #create a mask#################
    vertices = np.array([[(0,imshape[0]),(900, 600), (1000, 600), (imshape[1],imshape[0])]], dtype=np.int32)
    mask = np.zeros_like(edges)
    color = 255
    cv2.fillPoly(mask, vertices, color)
    ###############################

    maskedImg = cv2.bitwise_and(edges, mask)
    maskedIm3Channel = cv2.cvtColor(maskedImg, cv2.COLOR_GRAY2BGR)
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 45     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 40 #minimum number of pixels making up a line
    max_line_gap = 100    # maximum gap in pixels between connectable line segments
    lines = cv2.HoughLinesP(maskedImg, rho, theta, threshold, np.array([]),minLineLength=min_line_len, maxLineGap=max_line_gap)

    # Check if we got more than 1 line
    if lines is not None and len(lines) > 2:
        # Draw all lines onto image
        allLines = np.zeros_like(maskedImg)
        for i in range(len(lines)):
            for x1,y1,x2,y2 in lines[i]:
                cv2.line(allLines,(x1,y1),(x2,y2),(255,255,0),2) # plot line

        # # Plot all lines found
        # plt.figure(7)
        # plt.imshow(allLines,cmap='gray')
        # plt.title('All Hough Lines Found')

        #-----------------------Separate Lines Intro Positive/Negative Slope--------------------------
        # Separate line segments by their slope to decide left line vs. the right line
        slopePositiveLines = [] # x1 y1 x2 y2 slope
        slopeNegativeLines = []
        yValues = []

        # Loop through all lines
        addedPos = False
        addedNeg = False
        for currentLine in lines:
            # Get points of current Line
            for x1,y1,x2,y2 in currentLine:
                lineLength = ((x2-x1)**2 + (y2-y1)**2)**.5 # get line length
                if lineLength > 30: # if line is long enough
                    if x2 != x1: # dont divide by zero
                        slope = (y2-y1)/(x2-x1) # get slope line
                        if slope > 0:
                            # Check angle of line w/ xaxis. dont want vertical/horizontal lines
                            tanTheta = np.tan((abs(y2-y1))/(abs(x2-x1))) # tan(theta) value
                            ang = np.arctan(tanTheta)*180/np.pi
                            if abs(ang) < 85 and abs(ang) > 20:
                                slopeNegativeLines.append([x1,y1,x2,y2,-slope]) # add positive slope line
                                yValues.append(y1)
                                yValues.append(y2)
                                addedPos = True # note that we added a positive slope line
                        if slope < 0:
                            # Check angle of line w/ xaxis. dont want vertical/horizontal lines
                            tanTheta = np.tan((abs(y2-y1))/(abs(x2-x1))) # tan(theta) value
                            ang = np.arctan(tanTheta)*180/np.pi
                            if abs(ang) < 85 and abs(ang) > 20:
                                slopePositiveLines.append([x1,y1,x2,y2,-slope]) # add negative slope line
                                yValues.append(y1)
                                yValues.append(y2)
                                addedNeg = True # note that we added a negative slope line


        # If we didn't get any positive lines, go though again and just add any positive slope lines
        if not addedPos:
            for currentLine in lines:
                for x1,y1,x2,y2 in currentLine:
                    slope = (y2-y1)/(x2-x1)
                    if slope > 0:
                        # Check angle of line w/ xaxis. dont want vertical/horizontal lines
                        tanTheta = np.tan((abs(y2-y1))/(abs(x2-x1))) # tan(theta) value
                        ang = np.arctan(tanTheta)*180/np.pi
                        if abs(ang) < 80 and abs(ang) > 15:
                            slopeNegativeLines.append([x1,y1,x2,y2,-slope])
                            yValues.append(y1)
                            yValues.append(y2)

        # If we didn't get any negative lines, go through again and just add any negative slope lines
        if not addedNeg:
            for currentLine in lines:
                for x1,y1,x2,y2 in currentLine:
                    slope = (y2-y1)/(x2-x1)
                    if slope < 0:
                        # Check angle of line w/ xaxis. dont want vertical/horizontal lines
                        tanTheta = np.tan((abs(y2-y1))/(abs(x2-x1))) # tan(theta) value
                        ang = np.arctan(tanTheta)*180/np.pi
                        if abs(ang) < 85 and abs(ang) > 15:
                            slopePositiveLines.append([x1,y1,x2,y2,-slope])
                            yValues.append(y1)
                            yValues.append(y2)


        if not addedPos or not addedNeg:
            print('Not enough lines found')


        #------------------------Get Positive/Negative Slope Averages-----------------------------------
        # Average position of lines and extrapolate to the top and bottom of the lane.
        positiveSlopes = np.asarray(slopePositiveLines)[:,4]
        posSlopeMedian = np.median(positiveSlopes)
        posSlopeStdDev = np.std(positiveSlopes)
        posSlopesGood = []
        for slope in positiveSlopes:
        # if abs(slope-posSlopeMedian) < .9:
            if abs(slope-posSlopeMedian) < posSlopeMedian*.2:
                posSlopesGood.append(slope)
        posSlopeMean = np.mean(np.asarray(posSlopesGood))


        negativeSlopes = np.asarray(slopeNegativeLines)[:,4]
        negSlopeMedian = np.median(negativeSlopes)
        negSlopeStdDev = np.std(negativeSlopes)
        negSlopesGood = []
        for slope in negativeSlopes:
            if abs(slope-negSlopeMedian) < .9:
                negSlopesGood.append(slope)
        negSlopeMean = np.mean(np.asarray(negSlopesGood))

        #--------------------------Get Average x Coord When y Coord Of Line = 0----------------------------
        # Positive Lines
        xInterceptPos = []
        for line in slopePositiveLines:
                x1 = line[0]
                y1 = img.shape[0]-line[1] # y axis is flipped
                slope = line[4]
                yIntercept = y1-slope*x1
                xIntercept = -yIntercept/slope # find x intercept based off y = mx+b
                if xIntercept == xIntercept: # checks for nan
                    xInterceptPos.append(xIntercept) # add x intercept

        xIntPosMed = np.median(xInterceptPos) # get median
        xIntPosGood = [] # if not near median we get rid of that x point
        for line in slopePositiveLines:
                x1 = line[0]
                y1 = img.shape[0]-line[1]
                slope = line[4]
                yIntercept = y1-slope*x1
                xIntercept = -yIntercept/slope
                if abs(xIntercept-xIntPosMed) < .35*xIntPosMed: # check if near median
                    xIntPosGood.append(xIntercept)

        xInterceptPosMean = np.mean(np.asarray(xIntPosGood)) # average of good x intercepts for positive line

        # Negative Lines
        xInterceptNeg = []
        for line in slopeNegativeLines:
            x1 = line[0]
            y1 = img.shape[0]-line[1]
            slope = line[4]
            yIntercept = y1-slope*x1
            xIntercept = -yIntercept/slope
            if xIntercept == xIntercept: # check for nan
                    xInterceptNeg.append(xIntercept)

        xIntNegMed = np.median(xInterceptNeg)
        xIntNegGood = []
        for line in slopeNegativeLines:
            x1 = line[0]
            y1 = img.shape[0]-line[1]
            slope = line[4]
            yIntercept = y1-slope*x1
            xIntercept = -yIntercept/slope
            if abs(xIntercept-xIntNegMed)< .35*xIntNegMed:
                    xIntNegGood.append(xIntercept)

        xInterceptNegMean = np.mean(np.asarray(xIntNegGood))
        ############################3
    laneLines = np.zeros_like(edges)
    colorLines = img.copy()

    # Positive Slope Line
    slope = posSlopeMean
    x1 = xInterceptPosMean
    y1 = 0
    y2 = imshape[0] - (imshape[0]-imshape[0]*.35)
    x2 = (y2-y1)/slope + x1

    # Plot positive slope line
    x1 = int(round(x1))
    x2 = int(round(x2))
    y1 = int(round(y1))
    y2 = int(round(y2))
    cv2.line(laneLines,(x1,img.shape[0]-y1),(x2,imshape[0]-y2),(255,255,0),2) # plot line
    cv2.line(colorLines,(x1,img.shape[0]-y1),(x2,imshape[0]-y2),(0,255,0),4) # plot line on color image

    # Negative Slope Line
    slope = negSlopeMean
    x1N = xInterceptNegMean
    y1N = 0
    x2N = (y2-y1N)/slope + x1N

    # Plot negative Slope Line
    x1N = int(round(x1N))
    x2N = int(round(x2N))
    y1N = int(round(y1N))
    cv2.line(laneLines,(x1N,imshape[0]-y1N),(x2N,imshape[0]-y2),(255,255,0),2)
    cv2.line(colorLines,(x1N,img.shape[0]-y1N),(x2N,imshape[0]-y2),(0,255,0),4) # plot line on color iamge
    ########################
    # Plot lane lines
    plt.figure(8)
    plt.imshow(laneLines,cmap='gray')
    plt.title('Lane Lines')

    # Plot lane lines on original image
    plt.figure(9)
    plt.imshow(colorLines)
    plt.title('Lane Lines Color Image')

