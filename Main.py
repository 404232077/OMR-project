import cv2
import numpy as np
import utlis
import os
######################
# cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# while True:
#     ret, img = cam.read()
#     vis = img.copy()
#     cv2.imshow('getCamera', vis)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()

######################

path = "1.png" 
width = 700
height = 700
questions = 10
choice = 10
threshold_min = 1
threshold_max = 50
blur_kernel = 3
file_name = "student"
num = 0
csv_file = "save.csv"




######################

def empty(a):
    pass

cv2.namedWindow("image", 0)
cv2.createTrackbar("threshold", "image", 150, 255, empty)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    
    threshold_value = cv2.getTrackbarPos("threshold", "image")
    
    
    # read image##########################################################
    ret, img = cap.read()
    # img = cv2.imread(path)
    img = cv2.resize(img, (width, height))
    
    
    # preprocrssing#######################################################
    img_contour = img.copy()
    imgbiggestcontours = img.copy()
    img_save = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_blur = cv2.GaussianBlur(img_gray, (blur_kernel, blur_kernel), 1)
    img_canny = cv2.Canny(img_gray, threshold_min, threshold_max)
        
    try:
        # find contours#######################################################
        contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)    
        cv2.drawContours(img_contour, contours, -1, (0, 255, 0), 10)
        img_blank = np.zeros_like(img)
        
        # find rectangle
        rectCon = utlis.rectContour(contours)
        biggestContours = utlis.getCornerPoints(rectCon[0])
        print(biggestContours.size)
            
        if biggestContours.size != 0:
            cv2.drawContours(imgbiggestcontours, biggestContours, -1, (0, 255, 0), 20)
            biggestContours = utlis.reorder(biggestContours)
            
            pt1 = np.float32(biggestContours)
            pt2 = np.float32([[0, 0], [0, width], [height, 0], [width, height]])
            matrix = cv2.getPerspectiveTransform(pt1, pt2)
            imgWarpcolored = cv2.warpPerspective(img, matrix, (width, height))
            
            # apply threshold##################################################
            imgWarpgray = cv2.cvtColor(imgWarpcolored, cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgWarpgray, threshold_value, 255, cv2.THRESH_BINARY_INV)[1]
            # imgThresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            
            # 分割畫卡區(0為鉛筆)########################################################
            boxes, img_split = utlis.splitBoxes(imgThresh)
            img_split = cv2.resize(img_split, (width, height))
            mySelect = np.zeros((questions, choice))
            row = 0
            col = 0
            for c in boxes:
                totalPixels = cv2.countNonZero(c)
                mySelect[row][col] = totalPixels
                row += 1
                if row == choice:
                    col += 1
                    row = 0

            myIndex = []
            for x in range(questions - 1):
                question = mySelect[x]
                myIndexvalue = np.where(question == np.amax(question))
                if (myIndexvalue[0][0] + 1) == 10:
                    myIndex.append(0)
                else:
                    myIndex.append(myIndexvalue[0][0] + 1)

            myIndex = str(myIndex).replace(",","")
            myIndex = str(myIndex).replace(" ","")[1:-1]
            cv2.putText(img, f'student number {myIndex}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)

            
            if cv2.waitKey(1) & 0xFF == ord('s'):
                # Save CSV
                if os.path.isfile(csv_file):
                    file = open(csv_file, "a")
                    file.write(myIndex + "\n")
                    file.close()
                # Save file
                cv2.putText(img_save, "Saved", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)
                cv2.imshow("GG", img_save)
                cv2.imwrite(f'./save/{file_name}{str(num)}.png', img)
                cv2.waitKey(0)
                cv2.destroyWindow("GG")
                num += 1
            
                
    
            # show image ###########################################################
        
            img_array = ([img, img_split])
            img_stack = utlis.stackImages(img_array, 0.5)
            cv2.imshow("image", img_stack)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        
    except:
        img_array = ([img_blank, img_blank, img_blank, img_blank],
                      [img_blank, img_blank, img_blank, img_blank])
        img_stack = utlis.stackImages(img_array, 0.5)
        cv2.imshow("image", img_stack)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            break
    

cv2.destroyAllWindows()
cap.release()
