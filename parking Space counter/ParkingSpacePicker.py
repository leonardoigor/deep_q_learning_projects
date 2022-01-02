import cv2
import pickle

img = cv2.imread('parking Space counter\carParkImg.png')

width, height = 107, 48
try:
    with open('parking Space counter\CarParkPos', 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []


def mouseClick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        posList.append((x, y))
    if event == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            x1, y1 = pos
            if x1 < x < x1 + width and y1 < y < y1 + height:
                posList.pop(i)
                break


while True:
    img = cv2.imread('parking Space counter\carParkImg.png')

    for pos in posList:
        cv2.rectangle(img, pos, (pos[0]+width, pos[1]+height), (0, 255, 0), 2)
    cv2.imshow('img', img)
    cv2.setMouseCallback('img', mouseClick)
    cv2.waitKey(1)
