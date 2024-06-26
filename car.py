import pyautogui as py
import cv2
import numpy as np
import win32api , win32con , keyboard
import time

def click(x,y):
    win32api.setCursourPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN , 0,0)
    time.sleep(0.01)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0,0)

def accelerate(hold_time):
    start = time.time()
    while time.time() - start < hold_time:
        py.keyDown("w")
    py.keyup("w")

def turnLeft(hold_time):
    start = time.time()
    while time.time() - start < hold_time:
        py.keyDown("a")
    py.keyup("a")
    
def turnright(hold_time):
    start = time.time()
    while time.time() - start < hold_time:
        py.keyDown("d")
    py.keyup("d")
       
click(456,316)
accelerate(0.5)
turnLeft(0.3)
turnright(0.3)

def takescreenshot():
    im = py.screenshoot("screen.png" , region = (39,239,239,0) )
    img = np.array(im)
    img = cv2.cvtcolour(img,cv2.COLOUR_RGB2BGR)
    return img

def canny(img):
    gray = cv2.cvtColor(img,cv2.COLOUR_BGR2GRAY)
    Kernel = 5
    blur = cv2.GaussianBlur(gray, (Kernel,Kernel),0)
    canny = cv2.Canny(blur,50,150)
    return canny

def region_of_intrest(img):
    height = img.shape[0]
    width = img.shape[1]
    mask=np.zeros_like(img)
    car_mask = np.zeros_like(img)
    triangle = np.array([[(0,height),(width/2,height/2-50),(width,height-50)]],np.int32)
    car_triangle = np.array([[(width*1.0/4,height),(width/2,height/2),(width*3.0/4,height-50)]],np.int32)
    cv2.fillPoly(mask,triangle,255)
    cv2.fillPoly(mask ,car_triangle,0)
    masked_image = cv2.bitwise_and(img, mask)  
    return masked_image  

def houglines(img):
    houglines = cv2.HoughLinesP((img, 2,np.pi/180, 100,np.array([])),minLineLength = 40, maxlinegap = 10)
    return houglines

def make_points(img ,lineSI):
    slope, intercept = lineSI
    height = img.shape[0]
    y1 = int(height)
    y2 = int(y1*3.0/5)
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return[[x1,y1,x2,y2]]

def average_slope_intercept(img,lines):
    left_fit = []
    right_fit = []
    for line in lines:
        for x1, y1 , x2, y2 in line :
            fit = np.polyfit((x1,x2),(y1,y2),1)
            slope = fit[0]
            intercept = fit[1]

            if slope < 0 :
                left_fit.append((slope,intercept))
            else:
                right_fit.append((slope,intercept))
    left_fit_average = np.average(left_fit , axis = 0)
    right_fit_average = np.average(right_fit , axis=0)
    left_line = make_points(img , left_fit_average)
    right_line = make_points(img , right_fit_average)
    average_lines = [left_line , right_line]
    mLeft, lItercept = left_fit_average
    mRight,rIntercept = right_fit_average
    mLeft=round(mLeft,3)
    mRight=round(mRight,3)
    print(mLeft , mRight)
    if mLeft< -0.228:
        turnright(0.2)
        print("turnRight")
    if mRight >0.341 :
        turnLeft(0.2)
        turnLeft
    return average_lines

def display_lines_average(img, lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img , (x1,y1),(x2,y2),(0,0,255),10)
    return img

i = 0
click(456,316)
while i <50:
    accelerate(0.2)
    i+=1
    frame = takescreenshot()
    canny_image = canny(frame)
    m_img = region_of_intrest((canny_image))
    h_lines = houglines(m_img)
    avg_lines = average_slope_intercept(frame , h_lines)
    lines_image = display_lines_average(frame , avg_lines)
    cv2.imshow("win",lines_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
