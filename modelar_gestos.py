import numpy as np
import cv2

import random
import string

def get_random_string(length=8):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    print("Random string of length", length, "is:", result_str)
    return result_str

cap = cv2.VideoCapture(0)

while(True):
    ret, baseImage = cap.read()
    cv2.imshow('Selfie',baseImage)
    image_rescale = cv2.resize(baseImage, (128, 128))
    cv2.imwrite('./dataset_gestos/' + get_random_string()+ '.jpg' , image_rescale )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()