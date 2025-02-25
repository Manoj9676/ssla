import cv2 as cv
import numpy as np

import os
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
import pickle
from keras.models import load_model
from keras.applications import VGG16
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.models import model_from_json

old = None
vgg_model = load_model('model/model.h5')
class_labels = ['Speed limit (20km/h)','Speed limit (30km/h)','Speed limit (50km/h)','Speed limit (60km/h)','Speed limit (70km/h)',
'Speed limit (80km/h)','End of speed limit (80km/h)','Speed limit (100km/h)','Speed limit (120km/h)','No passing','Stop','No Entry',
'General caution','Traffic signals']


def cannyDetection(img):
    grayImg = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    blurImg = cv.GaussianBlur(grayImg, (5, 5), 0)
    cannyImg = cv.Canny(blurImg, 50, 150)
    return cannyImg

def segmentDetection(img):
    height = img.shape[0]
    polygons = np.array([[(0, height), (800, height), (380, 290)]])
    maskImg = np.zeros_like(img)
    cv.fillPoly(maskImg, polygons, 255)
    segmentImg = cv.bitwise_and(img, maskImg)
    return segmentImg

def calculateLines(frame, lines):
    left = []
    right = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_intercept = parameters[1]
        if slope < 0:
            left.append((slope, y_intercept))
        else:
            right.append((slope, y_intercept))
    left_avg = np.average(left, axis = 0)
    right_avg = np.average(right, axis = 0)
    left_line = calculateCoordinates(frame, left_avg)
    right_line = calculateCoordinates(frame, right_avg)
    return np.array([left_line, right_line])

def calculateCoordinates(frame, parameters):
    global old
    #print(str(parameters)+" "+str(type(parameters))+" "+str(np.isnan(parameters)))
    if old is None:
        old = parameters        
    if np.isnan(parameters.any()) == False:
        parameters = old
    slope, intercept = parameters
    y1 = frame.shape[0]
    y2 = int(y1 - 150)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def visualizeLines(frame, lines):
    lines_visualize = np.zeros_like(frame)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv.line(lines_visualize, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return lines_visualize

cap = cv.VideoCapture("Videos/input.mp4")
while (cap.isOpened()):
    ret, frame = cap.read()
    if frame is not None:
        canny = cannyDetection(frame)
        cv.imshow("cannyImage", canny)
        segment = segmentDetection(canny)
        hough = cv.HoughLinesP(segment, 2, np.pi / 180, 100, np.array([]), minLineLength = 100, maxLineGap = 50)
        if hough is not None:
            lines = calculateLines(frame, hough)
            linesVisualize = visualizeLines(frame, lines)
            cv.imshow("hough", linesVisualize)
            output = cv.addWeighted(frame, 0.9, linesVisualize, 1, 1)
        cv.imwrite("test.jpg",output)
        temps = cv.imread("test.jpg")
        h, w, c = temps.shape
        image = load_img("test.jpg", target_size=(80, 80))
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        (boxPreds, labelPreds) = vgg_model.predict(image)
        print(boxPreds)
        boxPreds = boxPreds[0]
        startX = int(boxPreds[0] * w)
        startY = int(boxPreds[1] * h)
        endX = int(boxPreds[2] * w)
        endY = int(boxPreds[3] * h)
        predict= np.argmax(labelPreds, axis=1)
        predict = predict[0]
        accuracy = np.amax(labelPreds, axis=1)
        print(str(class_labels[predict])+" "+str(accuracy))
        if accuracy > 0.97:
            cv.putText(output, "Recognized As "+str(class_labels[predict]), (startX, startY), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.rectangle(output, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv.imshow("output", output)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv.destroyAllWindows()
