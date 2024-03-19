#!/usr/bin/env python3
import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture(0)

def load_img(img_path):

    img = cv2.imread(img_path)
    height, width, channels = img.shape
    if(width > 1080):
        img = imutils.resize(img, width=1080)

    return img

def real_time():
    while(True):
        ret, img = cap.read()
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        corners = cv2.goodFeaturesToTrack(gray_image, 3000, qualityLevel=0.01, minDistance=3) 
        corners = np.float32(corners)
        for item in corners: 
            x, y = item[0] 
            x = int(x) 
            y = int(y) 
            cv2.circle(img, (x, y), 6, (0, 255, 0), -1) 
    
        # Showing the image 
        cv2.imshow('good_features', img) 
        # cv2.imshow("img",img)
        # cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break


def extract_feats(img_path):
    
    print("Extraction...")
    img = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)
    # while True:
    #     cv2.imshow('good_features', img) 
    #     cv2.waitKey(1)
    feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)

    corners = np.float32(feats)
    for item in corners: 
        x, y = item[0] 
        x = int(x) 
        y = int(y) 
        cv2.circle(img, (x, y), 6, (0, 255, 0), -1) 
    
        # Showing the image 
    cv2.imshow('good_features', img) 
    cv2.waitKey()

def detect_surface(img_main):
    # CONTOUR DETECTION
    img = cv2.cvtColor(img_main, cv2.COLOR_BGR2GRAY)
    # apply binary thresholding
    ret, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    cv2.imshow('binary', thresh)

    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    # print(type(contours)) # tuple
    print("Total contours: ",len(contours)) 
    print(contours[100])
    # draw contours on the original image
    image_copy = img_main.copy()


    if len(contours) != 0:
        cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(255, 0, 0), thickness=cv2.FILLED, lineType=cv2.LINE_AA)
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(image_copy,(x,y),(x+w,y+h),(0,255,0),10)


    # cv2.drawContours(img, contours, -1, color=(255, 255, 255), thickness=cv2.FILLED)
    # see the results
    cv2.imshow('None approximation', image_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def gpt_detect_horizontal_surfaces(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the contours
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Filter contours based on approximate rectangular shape
        if len(contour) >5:
            # Check if the contour is approximately horizontal
            print("contour: ",contour.shape )
            _, _, angle = cv2.fitEllipse(contour)
            # if angle > 85 and angle < 95:
            # Draw the contour on the original image
            cv2.drawContours(image, [approx], -1, (0, 255, 0), thickness=cv2.FILLED, lineType=cv2.LINE_AA)

    return image





if __name__ == "__main__":
    # real_time()
    # extract_feats("./table_2.jpg")
    img = load_img("./table_1.jpg")

    # gpt usage
    # out = gpt_detect_horizontal_surfaces(img)
    # cv2.imshow('Detected Horizontal Surfaces', out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # gpt2_detect_and_color_table("./table_2.jpg")

    detect_surface(img)




