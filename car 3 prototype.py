import cv2
import numpy as np
import keyboard
from PIL import ImageGrab
import time
import pyautogui
import warnings

def takeScreenshot():
    screenshot = ImageGrab.grab()
    return np.array(screenshot)

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def regionOfInterest(image):
    height = image.shape[0]
    width = image.shape[1]
    mask = np.zeros_like(image)
    polygon = np.array([[  # these are the vertices of the polygon
        (0, height),
        (width, height),
        (int(width / 2), int(height / 2))
    ]])
    cv2.fillPoly(mask, polygon, 255)
    cropped = cv2.bitwise_and(image, mask)
    return cropped

def houghLines(image):
    lines = cv2.HoughLinesP(image, 1, np.pi / 180, 100, np.array([]), minLineLength=100, maxLineGap=10)
    return lines

def averageSlopeIntercept(frame, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit([x1, x2], [y1, y2], 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(frame, left_fit_average)
    right_line = make_coordinates(frame, right_fit_average)
    return np.array([left_line, right_line])

def calculateSteeringAngle(frame, lines):
    height, width, _ = frame.shape
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", np.RankWarning)
                slope, intercept = np.polyfit([x1, x2], [y1, y2], 1)
            if slope < 0:
                x_mid = int((x1 + x2) / 2)
                y_mid = int((y1 + y2) / 2)
                cv2.circle(frame, (x_mid, y_mid), 5, (0, 255, 0), -1)
                steering_angle = np.arctan(slope) * 180 / np.pi
                return steering_angle
    return 0

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = int(image.shape[0])
    y2 = int(y1 * 3 / 5)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def displayLinesAverage(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 10)
    return line_image

def accelerate(value):
    pyautogui.press('up')

def turnLeft(value):
    pyautogui.press('left')

def turnRight(value):
    pyautogui.press('right')

def calculateSteeringAngle(frame, lines):
    height, width, _ = frame.shape
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            slope, intercept = np.polyfit([x1, x2], [y1, y2], 1)
            if slope < 0:
                x_mid = int((x1 + x2) / 2)
                y_mid = int((y1 + y2) / 2)
                cv2.circle(frame, (x_mid, y_mid), 5, (0, 255, 0), -1)
                steering_angle = np.arctan(slope) * 180 / np.pi
                return steering_angle
    return 0

def controlCar(steering_angle):
    if steering_angle > 10:
        turnRight(0.2)
    elif steering_angle < -10:
        turnLeft(0.2)
    else:
        accelerate(0.2)

def handleKeyPress(event):
    if event.name == 'up':
        accelerate(0.2)
    elif event.name == 'left':
        turnLeft(0.2)
    elif event.name == 'right':
        turnRight(0.2)

keyboard.on_press(handleKeyPress)

i = 0
pyautogui.click(456, 316)

while True:
   
       
    try:
        frame = takeScreenshot()
        canny_image = canny(frame)
        m_img = regionOfInterest(canny_image)
        h_lines = houghLines(m_img)
        avg_lines = averageSlopeIntercept(frame, h_lines)
        lines_image = displayLinesAverage(frame, avg_lines)
        steering_angle = calculateSteeringAngle(frame, avg_lines)
        controlCar(steering_angle)
        cv2.imshow("win", lines_image)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"An error occurred: {e}")