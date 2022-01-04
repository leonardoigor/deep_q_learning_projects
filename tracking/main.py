import cv2
print(cv2.__version__)
sourcer_video = 'tracking\mario.mp4'
sourcer_img = 'tracking\mario.png'

cap = cv2.VideoCapture(sourcer_video)

tracker = cv2.legacy.TrackerCSRT_create()

img = cv2.imread(sourcer_img)
box = cv2.selectROI('img', img, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()
print(box)

tracker.init(img, box)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        ok, box = tracker.update(frame)
        if ok:
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(
                box[0] + box[2]), int(box[1] + box[3])), (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
cv2.destroyAllWindows()
