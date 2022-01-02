import cv2
import pickle
import cvzone
import numpy as np

cap = None
try:
    with open('parking Space counter\CarParkPos', 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []


width, height = 107, 48

debugger = True


def Load():
    global cap
    cap = cv2.VideoCapture('parking Space counter\carPark.mp4')


def checkParkingSpace(img):
    for pos in posList:
        x, y = pos
        imgCrop = img[y:y+height, x:x+width]
        count = cv2.countNonZero(imgCrop)
        if debugger:
            cv2.imshow(str(x*y), imgCrop)
            # cvzone.putTextRect(frame, str(count), (x, y+height-10),
            #                    scale=1, thickness=1, offset=0)
        if count < 800:
            color = (0, 255, 0)
            tickness = 3
        else:
            color = (0, 0, 255)
            tickness = 2
        if debugger:
            cv2.rectangle(frame, pos, (x+width,  y+height),
                          color, tickness)


Load()
while(cap.isOpened()):
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    imGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(
        imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)

    imgMedian = cv2.medianBlur(imgThreshold, 5)
    imgDilated = cv2.dilate(imgMedian, np.ones((3, 3), np.uint8), iterations=1)

    if ret == True:
        checkParkingSpace(imgDilated)
        if debugger:
            cv2.imshow('Dilated', imgDilated)
        cv2.imshow('frame', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
