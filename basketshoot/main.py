import cv2 as cv
import time
from numpy import concatenate, array, zeros, polyfit
import threading as th
import os
import cvzone as zone
from cvzone.ColorModule import ColorFinder


videos = []
listDir = os.listdir("Videos")
colorFind = ColorFinder(False)

for i in listDir:
    videos.append(cv.VideoCapture("Videos/" + i))


pathN = 'Ball.png'
cv.namedWindow("TrackedBars")
cv.resizeWindow("TrackedBars", 640, 240)


minC, maxC = array([0, 49, 117]), array([58, 121, 255])
hsv = {'hmin': 0, 'smin': 150, 'vmin': 0, 'hmax': 18, 'smax': 244, 'vmax': 255}


X_, Y_ = [], []


def on_trackbar(v):
    global minC, maxC
    hue_min = cv.getTrackbarPos("Hue Min", "TrackedBars")
    hue_max = cv.getTrackbarPos("Hue Max", "TrackedBars")
    sat_min = cv.getTrackbarPos("Sat Min", "TrackedBars")
    sat_max = cv.getTrackbarPos("Sat Max", "TrackedBars")
    val_min = cv.getTrackbarPos("Val Min", "TrackedBars")
    val_max = cv.getTrackbarPos("Val Max", "TrackedBars")

    lower = array([hue_min, sat_min, val_min])
    upper = array([hue_max, sat_max, val_max])

    # minC = lower
    # maxC = upper


cv.createTrackbar("Hue Min", "TrackedBars", 0, 179, on_trackbar)
cv.createTrackbar("Hue Max", "TrackedBars", 179, 179, on_trackbar)
cv.createTrackbar("Sat Min", "TrackedBars", 0, 255, on_trackbar)
cv.createTrackbar("Sat Max", "TrackedBars", 255, 255, on_trackbar)
cv.createTrackbar("Val Min", "TrackedBars", 0, 255, on_trackbar)
cv.createTrackbar("Val Max", "TrackedBars", 255, 255, on_trackbar)


def getArea(img):
    percent = img.shape[0]*.875
    percent = int(percent)
    img = img[:percent, :, :]
    return img


def trackBall(img):
    mask = cv.inRange(img, minC, maxC)
    return cv.bitwise_and(img, img, mask=mask), mask


def findContours(img, mask):
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img, mask = colorFind.update(img, mask)
    contoursImg, contours = zone.findContours(img, mask, minArea=200)
    return contoursImg, contours


index = 0


def readAll():

    for i in listDir:
        videos.append(cv.VideoCapture("Videos/" + i))

    # for v in videos:
    #     v.read()


times = 1

xList = [ii for ii in range(0, 2000)]


def predict(img):

    # polinomial regression y=AX^2+BX+C

    A, B, C = polyfit(X_, Y_, 2)
    for x in xList:
        y = A*x**2 + B*x + C

        cv.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)


while True:
    ready, img = videos[index].read()
    if ready:
        img = cv.resize(img, (600, 600))
        img = getArea(img)
        img_orig = img.copy()
        # img, mask = trackBall(img)
        img, contours = findContours(img, hsv)
        if len(contours) > 0:
            cx, cy = contours[0]['center']
            X_.append(cx)
            Y_.append(cy)
        # img = cv.drawContours(img, contour, -1, (0, 255, 0), 3)
        for i, x in enumerate(X_):
            y = Y_[i]
            cv.circle(img_orig, (x, y), 5, (0, 255, 0), -1)

        if len(X_) > 2:
            predict(img_orig)
        cv.imshow('img', img)
        cv.imshow('img_orig', img_orig)
    if not ready:
        index = times % len(videos)
        times += 1
        readAll()
        X_, Y_ = [], []
        ready, img = videos[index].read()
    if cv.waitKey(100) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
