import sqlite3
import time
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import operator
import numpy as np



#for arrenging the contours in sequential order from left to right
def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]
 

    
# Load the classifier
clf = joblib.load("digits_cls.pkl")

#read an image
im = cv2.imread('12.jpg') 

#resizing an image
im = cv2.resize(im, (500,700)) 
#im=cv2.resize(im, (1300,700))

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray",im_gray)

im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
#cv2.imshow("blurring",im_gray)
    
# Threshold the image
ret,im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
#cv2.imshow("thresholding",im_th)
    
# Find contours in the image
_,ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
#sorting function call
ctrs=ctrs.sort(key=lambda x:get_contour_precedence(x, im_th.shape[1]))
    
# Get rectangles contains each contour    
rects = [cv2.boundingRect(ctr) for ctr in ctrs] 
    
def Segmentation_work(): 
    strFinalString=""
    for rect in reversed(rects):# variabe is rect and max iteration is rects
        # Draw the rectangles
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        # Make the rectangular region around the digit
        
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        
        # Resize the image
        
        roi = cv2.resize(im_th, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        #comparison and prediction of an image
        nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
        cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (25, 250, 250), 1)
        str1 = str(int(nbr[0]))
        strFinalString = str1 + strFinalString
       
    print(strFinalString)
    
    conn = sqlite3.connect('NPR.db_')
    c = conn.cursor() #creates new cursor object for executing SQL statements
    #c.execute("""CREATE TABLE VEHICLES(NUMBER_PLATE string, today TIME)""")
    #c.execute("INSERT INTO VEHICLES VALUES('123ABD', DATETIME('NOW'))")
    c.execute('INSERT INTO VEHICLES (NUMBER_PLATE, today ) VALUES (?, ?)', [ strFinalString, time.ctime()])
    c.execute("SELECT * FROM VEHICLES")
    print(c.fetchall())
    conn.commit()       #Commits the transactions
    conn.close()        #closes the connection
    

        
    cv2.imshow("Resulting Image with Rectangular ROIs", im)
    cv2.waitKey()
        
        
    #print(type(roi_hog_fd))
Segmentation_work()


   