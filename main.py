import cv2
import numpy as np

video = "http://192.168.1.104:4747/video"
cap = cv2.VideoCapture(video)
Img_width = 480
Img_height = 640
cap.set(3,Img_width)
cap.set(4,Img_height)

#圖片預處理
def preProcessing(frame):
    kernel = np.ones((5,5))
    gaussian = cv2.GaussianBlur(frame,(5,5),1)
    canny = cv2.Canny(gaussian,200,200)
    dilated = cv2.dilate(canny,kernel,iterations=2)
    eroded = cv2.erode(dilated,kernel,iterations=1)
    return eroded

#找出輪廓(畫面中最大的四邊形)
def getContours(frame):
    contours,hierarchy = cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    biggest = np.array([])
    maxArea = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 600:
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            obj_cor = len(approx)

            if obj_cor == 4 and area > maxArea:
                biggest = approx
                maxArea = area
        cv2.drawContours(frame_contours,biggest,-1,(255,0,0),20)

    return biggest


#找出視角轉換後的四個角
def reorder(points):
    points = points.reshape((4,2))
    newPoints = np.zeros((4,1,2),dtype='float32')
    add = points.sum(axis=1)
    newPoints[0] = points[np.argmin(add)]
    newPoints[3] = points[np.argmax(add)]
    diff = np.diff(points,axis=1)
    newPoints[1] = points[np.argmin(diff)]
    newPoints[2] = points[np.argmax(diff)]

    return newPoints

#視角轉換
def getWarp(frame,biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0],[Img_width,0],[0,Img_height],[Img_width,Img_height]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    output = cv2.warpPerspective(frame,matrix,(Img_width,Img_height))
    return output


while True:
    ret,frame = cap.read()
    frame_contours = frame.copy()

    res = preProcessing(frame)

    biggest = getContours(res)
    print(biggest.shape)
    print(biggest)

    if biggest.shape == (4,1,2):
        warped = getWarp(frame,biggest)
        cv2.imshow('warpedImage',warped)

    cv2.imshow('contours', frame_contours)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()