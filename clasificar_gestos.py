import cv2

import os


maximo = 817
max_train = 650
path  = './dataset_gestos/desconocido/'
for i, file_ in enumerate(os.listdir(path)):
    name, extension = os.path.splitext(file_)
    if extension == '.jpg':
        i = i+ 1
        if i < maximo:
            if i < max_train:
                os.rename(path+file_, path+'train/'+file_)
            
            #baseImage  = cv2.imread(path+ file_) 
            #image_rescale = cv2.resize(baseImage, (128, 128))
            #cv2.imwrite(path+ file_ , image_rescale )
            

        else:
            os.remove(path + file_)




