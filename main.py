import math
import random
import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Parameters
width, height = 1280, 720

# WebCam
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Hand Detector
detector = HandDetector(maxHands=1, detectionCon=0.8)


class SnakeGameClass:
    def __init__(self, pathFood):
        self.points = []  # all points in the snake
        self.lengths = []  # distance between each point
        self.currentLength = 0  # total length of the snake
        self.allowedLength = 150  # total allowed length
        self.previousHead = 0, 0  # previous head point coordinate
        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)  # Path of the food image
        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = 0, 0
        self.randomFoodLocation()
        self.score = 0
        self.gameOver = False

    def randomFoodLocation(self):
        self.foodPoint = random.randint(100, 1000), random.randint(100, 600)

    def update(self, imgMain, currentHead):

        if self.gameOver:
            cvzone.putTextRect(imgMain, "Game Over", [300, 400], scale=7, thickness=5, offset=20)
            cvzone.putTextRect(imgMain, f"Your Score is: {self.score}", [300, 550], scale=7, thickness=5, offset=20)
        else:

            px, py = self.previousHead
            cx, cy = currentHead

            self.points.append([cx, cy])
            distance = math.hypot(cx - px, cy - py)  # Finding distance from previous point to current point
            self.lengths.append(distance)
            self.currentLength += distance
            self.previousHead = cx, cy

            # Length reduction
            if self.currentLength > self.allowedLength:
                for i, length in enumerate(self.lengths):
                    self.currentLength -= length
                    self.lengths.pop(i)  # removing length as well from the lengths
                    self.points.pop(i)  # remove the points
                    if self.currentLength < self.allowedLength:
                        break

            # Check if Snake ate the food
            rx, ry = self.foodPoint
            centerOfFoodX, centerOfFoodY = self.wFood // 2, self.hFood // 2
            if rx - centerOfFoodX < cx < rx + centerOfFoodX and ry - centerOfFoodY < cy < ry + centerOfFoodY:
                self.randomFoodLocation()
                self.allowedLength += 50
                self.score += 1
                print(self.score)

            # Draw Snake
            if self.points:
                for i, point in enumerate(self.points):
                    if i != 0:
                        cv2.line(imgMain, self.points[i - 1], self.points[i], (0, 0, 255), 20)
                cv2.circle(imgMain, self.points[-1], 20, (200, 0, 200), cv2.FILLED)

            # Draw Food
            imgMain = cvzone.overlayPNG(imgMain, self.imgFood, (rx - self.wFood // 2, ry - self.hFood // 2))
            cvzone.putTextRect(imgMain, f"Score: {self.score}", [50, 80], scale=3, thickness=3, offset=10)

            # Check for Collision
            pts = np.array(self.points[:-2], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(imgMain, [pts], False, (0, 200, 0), 3)
            minDist = cv2.pointPolygonTest(pts, (cx, cy), True)

            if -0.2 <= minDist <= 0.1:  # Min distance to detect collision
                print("HIT")
                self.gameOver = True
                self.points = []  # all points in the snake
                self.lengths = []  # distance between each point
                self.currentLength = 0  # total length of the snake
                self.allowedLength = 150  # total allowed length
                self.previousHead = 0, 0  # previous head point coordinate
                self.randomFoodLocation()
                self.score = 0

        return imgMain


game = SnakeGameClass("Donut.png")

while True:
    # Get the frame from the webcam
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Get the hands
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        lmList = hands[0]['lmList']
        pointIndex = lmList[8][0:2]  # 8 is the point located at the tip of the Index finder
        img = game.update(img, pointIndex)

    # Display
    cv2.imshow("Snake Game", img)
    key = cv2.waitKey(1)

    if key == ord('r'):
        game.gameOver = False
