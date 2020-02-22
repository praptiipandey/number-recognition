import sys
import cv2
import sqlite3
import time

from sklearn.externals import joblib
from skimage.feature import hog
#import classifier 

#from sklearn.externals import joblib
#from skimage.feature import hog
import numpy as np

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtWidgets import QDialog,QApplication,QFileDialog
from PyQt5.uic import loadUi

s=""
class life2coading(QDialog):
    

    def __init__(self):
        super(life2coading,self).__init__()
        loadUi('life2coading.ui',self)
        self.image=None
        self.loadButton.clicked.connect(self.loadClicked)
        #self.trainButton.clicked.connect(self.TrainClicked)
        self.segmentButton.clicked.connect(self.SegmentClicked)
        self.recogniseButton.clicked.connect(self.RecogniseClicked)
        
  
        
    @pyqtSlot()
    def loadClicked(self):
        fname,filter = QFileDialog.getOpenFileName(self,'Open File','C:\\',"Image Files (*.jpg)")
        if fname:
            self.loadImage(fname)
        else:
            print('Invaled image')   
            
            
    def loadImage(self,fname):
        self.img=cv2.imread(fname,cv2.IMREAD_COLOR)
        
        i=1
        img_name="image{}.jpg".format(i)
        cv2.imwrite(img_name, self.img)
        
        self.image=cv2.resize(self.img,(int(self.img.shape[1]/2), int(self.img.shape[0]/2)))
        self.displayImage()
        
                                
    def displayImage(self):
        qformat=QImage.Format_Indexed8
            
        if len(self.image.shape)==3:
            if(self.image.shape[2])==4:
                qformat=QImage.Format_RGBA8888
            else:
                qformat=QImage.Format_RGB888  
            
        img=QImage(self.image,self.image.shape[1],self.image.shape[0],self.image.strides[0],qformat)
        img= img.rgbSwapped()
        
        
        self.imgLabel1.setPixmap(QPixmap.fromImage(img))
        self.imgLabel1.setAlignment(QtCore.Qt.AlignCenter|QtCore.Qt.AlignVCenter)     
    
    #@pyqtSlot()
    #def TrainClicked(self):
        #classifier.TrainingMachine()            
        
    @pyqtSlot()
    def SegmentClicked():
                
        def get_contour_precedence(contour, cols):
            tolerance_factor = 10
            origin = cv2.boundingRect(contour)
            return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]
            
        # Load the classifier
        clf = joblib.load("digits_cls.pkl")

        #read an image
        im = cv2.imread('12.jpg') 

        #resizing an image
        #im = cv2.resize(im, 0) 
        im=cv2.resize(im, (1300,700))
        
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
        ctrs.sort(key=lambda x:get_contour_precedence(x, im_th.shape[1]))
        
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
                roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
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
        Segmentation_work()

        
    @pyqtSlot()
    def RecogniseClicked(self):
        import predict
        predict.predict_image()    
        

app=QApplication(sys.argv)
window=life2coading()
window.setWindowTitle('Number Recogniton')
window.show()
sys.exit(app.exec_())

    
                