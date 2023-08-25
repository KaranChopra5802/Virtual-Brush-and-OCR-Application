import cv2
import os
import warnings
import numpy as np
import pytesseract
import HandTrackingModule as htm
import pygame

brushThickness = 15
eraserThickness = 75
score = 0
keyFinal = 0
press = True

with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)

pygame.init()
correct_sound = pygame.mixer.Sound('C:\\Users\\Karan\\Desktop\\Projects\\Brush Detector\\Sounds\\correct_sound.mp3')
wrong_sound = pygame.mixer.Sound('C:\\Users\\Karan\\Desktop\\Projects\\Brush Detector\\Sounds\\wrong_sound.mp3')

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


folderPath = "Header"
myList = os.listdir(folderPath)

folderPath1 = "Button"
myList1 = os.listdir(folderPath1)

folderPath2 = "Letter"
myList2 = os.listdir(folderPath2)

overlayList = []
overlayList1 = []
overlayList2 = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
header = overlayList[0]
detectedText = overlayList[2]
footer = overlayList[3]
scoreCard = overlayList[4]

for imPath in myList1:
    image1 = cv2.imread(f'{folderPath1}/{imPath}')
    overlayList1.append(image1)
button = overlayList1[0]

for imPath in myList2:
    image2 = cv2.imread(f'{folderPath2}/{imPath}')
    overlayList2.append(image2)
letter = overlayList2[0]

drawColor = (0, 0, 255)
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = htm.handDetector(detectionConf=0.85)
xp, yp = 0, 0

imgCanvas = np.zeros((480, 640, 3), np.uint8)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    scoreText = str(score)
    cv2.putText(img, scoreText, (17, 245), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:

        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()

        letterDisplay = 0

        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            print(chr(key))
            key = key - 32
            keyFinal = key
            press = True
            letterDisplay = key - 65
            letter = overlayList2[letterDisplay+1]

        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            if 0 < x1 < 60 and 90 < y1 < 150:

                button = overlayList1[0]
                detected_text = pytesseract.image_to_string(imgInv, config='--psm 10')
                detected_text = detected_text[0]
                if detected_text != 'o':
                    print(detected_text)
                detectedTextInt = ord(detected_text)
                if detectedTextInt == keyFinal or detectedTextInt-32 == keyFinal:
                    if press:
                        score = score + 1
                        scoreText = str(score)
                        press = False
                        correct_sound.play()
                        letter = overlayList2[0]
                        cv2.rectangle(imgCanvas, (0, 0), (640, 360), (0, 0, 0), cv2.FILLED)
                else:
                    if press:
                        press = False
                        wrong_sound.play()
                        letter = overlayList2[0]
                        cv2.rectangle(imgCanvas, (0, 0), (640, 360), (0, 0, 0), cv2.FILLED)

                cv2.putText(img, detected_text, (582, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            if y1 < 90:
                if 0 < x1 < 320:
                    header = overlayList[0]
                    drawColor = (0, 0, 255)
                elif 320 < x1 < 640:
                    header = overlayList[1]
                    drawColor = (0, 0, 0)

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
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
    img[170:210, 0:60] = scoreCard
    img[100:150, 460:568] = detectedText
    img[164:340, 460:624] = letter
    img[360:490, 0:640] = footer
    cv2.rectangle(img, (70, 100), (450, 340), (0, 0, 0), 2)

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    cv2.imshow("Result", img)
    cv2.waitKey(1)
