import cv2 as cv
import time
from numpy import concatenate, array, zeros
import threading as th


done = True
img = ''
debugger = True
grayimg, blurimg, cannyImg, imgCopyC = zeros((480, 640, 3)),  zeros(
    (480, 640, 3)),  zeros((480, 640, 3)),  zeros((480, 640, 3))
kernel1 = 190
kernel2 = 10
roi = zeros((200, 400, 3))


imgCopy = zeros((480, 640, 3))


def getImg():
    global img
    img = 'basketshoot/teste.png'

    img = cv.imread(img)
    height, width = (480, 640)
    img = cv.resize(img, (width, height))


def getFps(last_time):
    fps = 1 / (time.time() - last_time)
    return fps, time.time()


def getContours(img):
    global grayimg, blurimg, cannyImg, imgCopyC, kernel1, kernel2, done, imgCopy, roi
    done = False
    imgCopy = img.copy()
    grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurimg = cv.GaussianBlur(grayimg, (5, 5), 1)
    cannyImg = cv.Canny(blurimg, kernel1, kernel2)
    countours, _ = cv.findContours(
        cannyImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    imgCopyC = grayimg.copy()
    imgCopyC = cv.drawContours(imgCopyC, countours, -1, (0, 255, 0), 3)
    maxArea = 0
    maxAdges = []
    for c in countours:
        boundingBox = cv.boundingRect(c)
        roi = imgCopy[boundingBox[1]:boundingBox[1] + boundingBox[3],
                      boundingBox[0]:boundingBox[0] + boundingBox[2]]

        area = cv.contourArea(c)
        if area > 100:
            peri = cv.arcLength(c, True)
            adges = cv.approxPolyDP(c, 0.02*peri, True)
            if area > maxArea and len(adges) == 4:
                maxArea = area
                maxAdges = adges
    if len(maxAdges) != 0:
        cv.drawContours(imgCopy, maxAdges, -1, (0, 255, 0), 10)
    roi = cv.resize(roi, (200, 400))
    done = True


cap = cv.VideoCapture(0)
started_time = time.time()
while True:
    getImg()
    th1 = th.Thread(target=getContours, args=(img,))
    fps, started_time = getFps(started_time)
    # cv.putText(img, 'FPS: ' + str(fps), (10, 30),
    #            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    concat = concatenate((img, imgCopy), axis=1)
    cv.imshow('stack', concat)
    if debugger:
        stack = concatenate((grayimg, blurimg, cannyImg, imgCopyC), axis=1)
        cv.imshow('stackDebugger', stack)
    cv.imshow('roi', roi)

    # kernel1 = input("kernel1 ")  # Python 3
    # kernel1 = int(kernel1)
    # kernel2 = input("kernel2 ")  # Python 3
    # kernel2 = int(kernel2)
    if done:
        th1.start()
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
