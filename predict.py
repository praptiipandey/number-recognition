import numpy as np
import cv2
from sklearn.externals import joblib
import matplotlib.pyplot as plt

def predict_image():
    img=cv2.imread("12.jpg")
    clf=joblib.load("digit_cls.pkl")
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    ret, img = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY_INV)

    cv2.imwrite("out.jpg",img)

    #img=cv2.resize(img,(8,8))
   # plt.imshow(img)

    #img=img.flatten()
    print('Prediction:',clf.predict(img))


#x=[0,1,2,3,4,5,6,7,8,9]
#plt.bar(x,y[0])
#plt.show()
#print(clf.predict(img))

#plt.imshow(img)




cv2.waitKey(0)
cv2.destroyAllWindows()
predict_image()