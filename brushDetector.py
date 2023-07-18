import cv2
import numpy as np
import os

import pytesseract

import HandTrackingModule as htm

brushThickness = 15
eraserThickness = 75

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)

folderPath1 = "Button"
myList1 = os.listdir(folderPath1)
print(myList1)

overlayList = []
overlayList1 = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
header = overlayList[0]

for imPath in myList1:
    image1 = cv2.imread(f'{folderPath1}/{imPath}')
    overlayList1.append(image1)
button = overlayList1[0]

drawColor = (0, 255, 0)
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 360)

detector = htm.handDetector(detectionConf=0.85)
xp, yp = 0, 0

imgCanvas = np.zeros((360, 640, 3), np.uint8)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:

        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()


        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection mode")
            if 0 < x1 < 60 and 90 < y1 < 150:
                button = overlayList1[0]
                detected_text = pytesseract.image_to_string(imgInv, config='--psm 10')
                print("Text detected :" + detected_text)
                cv2.putText(img, detected_text, (530, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if y1 < 90:
                if 0 < x1 < 320:
                    header = overlayList[0]
                    drawColor = (0, 255, 0)
                elif 320 < x1 < 640:
                    header = overlayList[1]
                    drawColor = (0, 0, 0)

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1

    img[0:90, 0:640] = header
    img[90:150, 0:60] = button

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    cv2.imshow("Result", img)
    #cv2.imshow("Canvas", imgInv)
    cv2.waitKey(1)
