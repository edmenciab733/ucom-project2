import numpy as np
import cv2
import math

import random
import string

lower_color = np.array([0,20,70], dtype=np.uint8)
upper_color = np.array([20,255,255], dtype=np.uint8)

def get_random_string(length=8):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    print("Random string of length", length, "is:", result_str)
    return result_str

cap = cv2.VideoCapture(0)

while(True):
    try:
        ret, baseImage = cap.read()
        
        roi=baseImage[50:500, 50:500]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_color, upper_color)
        cv2.rectangle(baseImage,(50,50),(500,500),(0,255,0),0)  
        print(cv2.countNonZero(mask))
        if cv2.countNonZero(mask) > 30636:
            print("Verificar gesto")

    except:
        pass

    cv2.imshow('Selfie',baseImage)
    cv2.imshow('Gesto',mask)
    
   
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()