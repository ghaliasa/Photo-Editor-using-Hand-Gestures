import cv2 as cv
import math
from pynput.mouse import Controller
from tkinter.filedialog import askopenfilename, asksaveasfile
import numpy as np
from pynput.mouse import Button as bt
from tkinter import colorchooser
from base64 import b16encode

main_image = cv.imread("Choose  Image.jpg")
rotate_angle = 0
wcam, hcam = 1500, 1000
mouse = Controller()
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)
font = cv.FONT_HERSHEY_SIMPLEX
rotationButton = False
scale_button = True
translate_button = False
release = False

scaleinorout = 0

extTop = (0, 0)
lasttop = (0, 0)
args = []
extLeft = 0
extRight = 0
extBot = 0
extRight_list = []
extLeft_list = []
extTop_list = []
extBot_list = []
pointsxystart = []
pointsxycountorr = []
last_Ten_Finger_Counts = [0, 0, 0, 0]
length = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
finger_count = 0
point_Start = 0
len1 = 0
arga = 0
arg = 0
color_code = ((0, 0, 0), "kokok")

##############################
curr_len = 0
prev_len = 0

diffx = 0
diffy = 0

curr_far = (0, 0)
prev_far = (0, 0)

mouse_x = 50
mouse_y = 50

save_bool = False
fresh = (0, 0)

frame1 = 0
frame2 = 0


##############################


def chooseImage(x):
    global main_image
    image_path = askopenfilename()
    main_image = cv.imread(image_path)


def rotate(angel):
    global main_image
    rows, cols, _ = main_image.shape
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), angel, 1)
    main_image = cv.warpAffine(main_image, M, (cols, rows))


def scale_in():
    global main_image
    dimensions = main_image.shape
    if dimensions[0] <= 0 and dimensions[1] <= 0:
        return
    if int(dimensions[0] / 20) > 0 and int(dimensions[1] / 20) > 0:
        main_image = cv.resize(main_image, (int(dimensions[1] / 20), int(dimensions[0] / 20)))


def scale_out():
    global main_image
    dimensions = main_image.shape
    if dimensions[0] <= 0 and dimensions[1] <= 0:
        return
    if 700 > int(dimensions[0] * 2) > 0 and 700 > int(dimensions[1] * 2) > 0:
        main_image = cv.resize(main_image, (int(dimensions[1] * 2), int(dimensions[0] * 2)))


def translate(x, y):
    global main_image
    rows, cols, _ = main_image.shape
    M = np.float32([[1, 0, x], [0, 1, y]])
    main_image = cv.warpAffine(main_image, M, (cols, rows))


def save():
    global main_image
    cv.imwrite("output.jpg", main_image)


def skew(x, y, angle):
    global main_image
    gray_img = cv.cvtColor(main_image, cv.COLOR_BGR2GRAY)
    gray_img = cv.bitwise_not(gray_img)
    coordinates = np.column_stack(np.where(gray_img > 0))
    ang = cv.minAreaRect(coordinates)[-1]
    if ang < -45:
        ang = -(90 + ang)
    else:
        ang = -ang
    height, width = main_image.shape[:2]
    center_img = (width / 2, height / 2)
    rotationMatrix = cv.getRotationMatrix2D(center_img, angle, 1.0)
    main_image = cv.warpAffine(
        main_image,
        rotationMatrix,
        (width + x, height + y),
        borderMode=cv.BORDER_REFLECT
    )


def warp(top_y, bottom_y, left_x, right_x):
    global main_image
    rows, cols, _ = main_image.shape
    if rows > 500 and cols > 350:
        pts1 = np.float32(
            [
                [left_x, bottom_y],
                [right_x, bottom_y],
                [left_x, top_y],
                [right_x, top_y]
            ]
        )
        width, height = rows, cols
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        matrix = cv.getPerspectiveTransform(pts1, pts2)
        main_image = cv.warpPerspective(main_image, matrix, (width, height))


def choose_color(x):
    global color_code
    color_code = colorchooser.askcolor(title="Choose color")


cv.namedWindow('image')
cv.createTrackbar("Choose Image", 'image', 0, 1, chooseImage)
cv.createTrackbar("Choose Color", 'image', 0, 1, choose_color)

while 1:
    cv.imshow('image', main_image)
    _, frame = cap.read()
    img_YCrCb = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
    YCrCb_mask = cv.inRange(img_YCrCb, (0, 135, 85), (255, 180, 135))
    YCrCb_mask = cv.morphologyEx(
        YCrCb_mask,
        cv.MORPH_CLOSE,
        np.ones((3, 3),
                np.uint8)
    )
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
        for i in range(x, x + w):
            for j in range(y, y + h):
                YCrCb_mask[j][i] = 0
    YCrCb_mask = cv.medianBlur(YCrCb_mask, 9)
    kernel = np.ones((5, 5), np.uint8)
    YCrCb_mask = cv.dilate(YCrCb_mask, kernel, iterations=3)
    YCrCb_mask = cv.morphologyEx(YCrCb_mask, cv.MORPH_OPEN, kernel)
    YCrCb_result = cv.bitwise_not(YCrCb_mask)
    contours, hierarchy = cv.findContours(
        YCrCb_result,
        cv.RETR_TREE,
        cv.CHAIN_APPROX_SIMPLE
    )
    if len(contours) >= 2:
        cnt = max(contours[1:], key=lambda x: cv.contourArea(x))
        epsilon = 0.0005 * cv.arcLength(cnt, True)
        data = cv.approxPolyDP(cnt, epsilon, True)
        hull = cv.convexHull(cnt)
        cv.drawContours(frame, [cnt], -1, (50, 50, 150), 2)
        cv.drawContours(frame, [hull], -1, (0, 255, 0), 2)

        extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
        extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
        lasttop = extTop
        extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
        extBot = tuple(cnt[cnt[:, :, 1].argmax()][0])
        #
        # cv.circle(frame, extLeft, 8, (0, 0, 255), -1)
        # cv.circle(frame, extRight, 8, (0, 255, 0), -1)
        # cv.circle(frame, extTop, 8, (255, 0, 0), -1)
        # cv.circle(frame, extBot, 8, (255, 255, 0), -1)

        areahull = cv.contourArea(hull)
        areacnt = cv.contourArea(cnt)
        arearatio = ((areahull - areacnt) / areacnt) * 100
        hull = cv.convexHull(cnt, returnPoints=False)
        defects = cv.convexityDefects(cnt, hull)
        len1 = 0
        if defects is not None:
            l = 0
            x1, y1 = 0, 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(cnt[s][0])
                x1, y1 = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                x2, y2 = tuple(cnt[e][0])
                far = tuple(cnt[f][0])

                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                s = (a + b + c) / 2
                ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

                length[1] = length[0]
                length[0] = math.hypot(x2 - x1, y2 - y1)

                prev_len = curr_len
                curr_len = math.dist(start, end)

                prev_far = curr_far
                curr_far = far

                d = (2 * ar) / a

                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

                if angle <= 90 and d > 30:
                    l += 1
                    cv.circle(frame, start, 15, [255, 255, 0], -1)
                    cv.circle(frame, end, 15, [0, 255, 0], -1)
                    cv.circle(frame, far, 15, [0, 0, 255], -1)
                    # cv.circle(frame, far, 5, [0, 255, 255], -1)
                    fresh = far

                cv.line(frame, start, end, [0, 255, 0], 2)
            l += 1

            if l == 1:
                frame1 = 1
                if areacnt < 2000:
                    cv.putText(frame, 'Put hand in the box', (0, 50),
                               font, 2, (0, 0, 255), 3, cv.LINE_AA)
                else:
                    if arearatio < 12:
                        cv.putText(frame, '0', (0, 50), font, 2,
                                   (0, 0, 255), 3, cv.LINE_AA)
                        print("save")
                        save()
                    else:
                        cv.putText(frame, '1', (0, 50), font, 2,
                                   (0, 0, 255), 3, cv.LINE_AA)
                        if lasttop[0] < extTop[0]:
                            print("rotate right")
                            rotate(90)
                        else:
                            print("rotate left")
                            rotate(-90)
                        if lasttop[1] < extTop[1]:
                            print("skew")
                            skew(
                                extTop[0] - lasttop[0],
                                extTop[1] - lasttop[1],
                                angle
                            )
                        else:
                            print("warp")
                            warp(
                                top_y=extTop[1],
                                bottom_y=extBot[1],
                                left_x=extLeft[0],
                                right_x=extRight[0],
                            )
            elif l == 2:
                frame2 = 2
                cv.putText(frame, '2', (0, 50), font, 2,
                           (0, 0, 255), 3, cv.LINE_AA)
                if not release:
                    mouse_x = x1
                    mouse_y = y1
                    mouse.position = (mouse_x, mouse_y)
                    if length[0] < 5:
                        mouse.press(bt.left)
                    if length[0] > 6:
                        mouse_x = 150
                        mouse_y = 150
                        mouse.release(bt.left)

            elif l == 3:
                if arearatio < 27:
                    cv.putText(frame, '3', (0, 50), font, 2,
                               (0, 0, 255), 3, cv.LINE_AA)
                    print("translate")
                    if prev_far[0] < curr_far[0] and prev_far[1] < curr_far[1]:
                        translate(curr_far[0], curr_far[1])
                    else:
                        translate(-curr_far[0], -curr_far[1])
            elif l == 4:
                cv.putText(frame, '4', (0, 50), font, 2,
                           (0, 0, 255), 3, cv.LINE_AA)
                cv.line(frame, start, end, [255, 255, 255], 20)
                if scale_button:
                    if prev_len < curr_len:
                        print("scale_in")
                        scale_in()
                        scale_button = False
                    else:
                        print("scale_out")
                        scale_out()
                        scale_button = False
                else:
                    scale_button = True
            elif l == 5:
                cv.putText(frame, '5', (0, 50), font, 2,
                           (0, 0, 255), 3, cv.LINE_AA)
                cv.line(
                    main_image,
                    curr_far,
                    prev_far,
                    [color_code[0][2],
                     color_code[0][1],
                     color_code[0][0]],
                    15
                )
            else:
                cv.putText(frame, 'reposition', (10, 50), font,
                           2, (0, 0, 255), 3, cv.LINE_AA)
    # cv.imshow("YCrCb_mask", YCrCb_mask)
    # cv.imshow("YCrCb_result", YCrCb_result)
    cv.imshow("Frame", frame)
    print(color_code)
    if cv.waitKey(1) == ord("q"):
        release = True
        mouse.release(bt.left)

    if cv.waitKey(1) == ord("p"):
        save_bool = not save_bool

    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()
