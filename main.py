import numpy as np
import cv2
import dlib
import threading



import random
import string


from keras.models import load_model
from keras.preprocessing import image

FPS = 30
rectangleColor = (55,165,255)
currentFaceID = 0
faceNames = {}


# Define the upper and lower boundaries for a color to be considered "Blue"
lower_color = np.array([0,20,70], dtype=np.uint8)
upper_color = np.array([20,255,255], dtype=np.uint8)

# Define a 5x5 kernel for erosion and dilation
kernel = np.ones((5, 5), np.uint8)

cap = cv2.VideoCapture(0)

faceTrackers = {}
keypointTrackers = {}


def doRecognizePerson(faceNames, fId):
    faceNames[ fId ] = "Person: "+ str(fId)

def showMouth():
    if show_mouth: 
        show_mouth = False
    else:
        show_mouth = True
    print(show_mouth)
def get_random_string(length=8):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    print("Random string of length", length, "is:", result_str)
    return result_str

face_cascade = cv2.CascadeClassifier('./clasificadores/haarcascade_frontalface_alt.xml')
if face_cascade.empty(): raise Exception("¿Está seguro que es la ruta correcta?")

my_model= load_model("./keypoints.h5")
my_model_gestos= load_model("./gestos.h5")
tongue  = cv2.imread("./lengua.jpg") 
mouth = cv2.imread("./boca.png") 
eyes_left  = cv2.imread("./izquierdo.png")
rigth_left  = cv2.imread("./derecho.png") 
nose  = cv2.imread("./nose.jpg") 
show_mouth = False
show_nose = False
show_point = False

frameCounter = 0


while(True):
    # Capture frame-by-frame
    ret, baseImage = cap.read()
    resultImage = baseImage.copy()
    cv2.putText(baseImage, "FPS : " + str(int(FPS)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
    frameCounter += 1
    #Increase the currentFaceID counter
    fidsToDelete = []
    for fid in list(faceTrackers):
        trackingQuality = faceTrackers[ fid ].update( baseImage )
        #If the tracking quality is not good enough, we must delete
        #this tracker
        if trackingQuality < 7:
            fidsToDelete.append( fid )
           
        for fid in fidsToDelete:
            print("Removing face id " + str(fid) + " from list of trackers")
            faceTrackers.pop( fid , None )  
    if (frameCounter % 2) == 0:
        grayImage = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(grayImage, 1.3, 5)
        if(len(faces) > 0):
            faces = faces.astype('int32') 
        
        for (x, y, w, h) in faces:
            #calculate the centerpoint
            x_bar = x + 0.5 * w
            y_bar = y + 0.5 * h
            matchedFaceId = None

            for fid in list(faceTrackers):
                    tracked_position =  faceTrackers[fid].get_position()
                    #tracked_position = tracked_position.astype('int32')
                    t_x = int(tracked_position.left())
                    t_y = int(tracked_position.top())
                    t_w = int(tracked_position.width())
                    t_h = int(tracked_position.height())


                    #calculate the centerpoint
                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h

                    #check if the centerpoint of the face is within the 
                    #rectangleof a tracker region. Also, the centerpoint
                    #of the tracker region must be within the region 
                    #detected as a face. If both of these conditions hold
                    #we have a match
                    if ( ( t_x <= x_bar   <= (t_x + t_w)) and  ( t_y <= y_bar   <= (t_y + t_h)) and  ( x   <= t_x_bar <= (x   + w  )) and  ( y   <= t_y_bar <= (y   + h  ))):
                        matchedFaceId = fid


                    #If no matched fid, then we have to create a new tracker
            if matchedFaceId is None:

                print("Creating new tracker " + str(currentFaceID))

                #Create and store the tracker 
                tracker = dlib.correlation_tracker()
                tracker.start_track(baseImage, dlib.rectangle( x-10,  y-20, x+w+10,  y+h+20))

                faceTrackers[ currentFaceID ] = tracker

                #Start a new thread that is used to  
                #face recognition. 
                detected_face = baseImage[int(y):int(y+h), int(x):int(x+w)] #crop detected face
                detected_face = cv2.resize(detected_face, (224, 224)) #resize to 224x224
                detected_face= cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB)
                originalImageBase =  cv2.cvtColor(baseImage, cv2.COLOR_BGR2RGB)


                # Convert to HSV and GRAY for convenience
                hsv = cv2.cvtColor(originalImageBase, cv2.COLOR_BGR2HSV)
                gray = cv2.cvtColor(originalImageBase, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(gray, (96, 96), interpolation = cv2.INTER_AREA)
                face_resized = face_resized.reshape(1, 96, 96, 1)
                keypoints = np.squeeze(my_model.predict(face_resized))
                
                keypointTrackers[ currentFaceID ] = keypoints                

                t = threading.Thread( target = doRecognizePerson ,
                                           args=(faceNames, currentFaceID ))
                t.start()

                currentFaceID += 1

    

    for fid in list(faceTrackers):
        tracked_position =  faceTrackers[fid].get_position()
        keypoint = keypointTrackers[ fid ] 

        t_x = int(tracked_position.left())
        t_y = int(tracked_position.top())
        t_w = int(tracked_position.width())
        t_h = int(tracked_position.height())


        alpha = 0.4

        layer = np.ones((96, 96, 3),  dtype=np.uint8) * 255
        layerMouth= np.ones((96, 96, 3),  dtype=np.uint8) * 255

       
        #layer = resultImage[t_y:t_y+t_h, t_x:t_x+t_w]
        #for j in range(1, len(keypoint),2):
        #   cv2.circle(layer, (keypoint[j -1],keypoint[j]) , 1, (0,255,0), 1)

        layer = cv2.resize(layer,(int(t_w),int(t_h)))


        #layer = cv2.cvtColor(layer, cv2.COLOR_BGR2BGRA)
        #cv2.imwrite('image.jpg', layer )
        

        if fid in list(faceNames):

            try:
                roi=resultImage[50:500, 50:500]
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower_color, upper_color)
                cv2.rectangle(resultImage,(50,50),(500,500),(0,255,0),0)  
                #print(cv2.countNonZero(mask) )
                if cv2.countNonZero(mask) > 30636:

                    msg=resultImage[50:500, 50:500]
                    msg = cv2.resize(msg,(int(64),int(64)))

                   # cv2.imwrite('./dgestos_aux/' + get_random_string() + '.jpg' , msg )
                    msg = msg / 255.0
                    msg = np.expand_dims(msg, axis = 0)
                    show_point = False
                    show_mouth = False
                    show_nose  = False
                    result = my_model_gestos.predict( msg)
                    if np.argmax(result[0]) == 0:
                        show_point = True
                    elif np.argmax(result[0]) == 1:
                        show_mouth = True
                    elif np.argmax(result[0]) == 3:
                        show_nose  = True
                    elif np.argmax(result[0]) == 2:
                        show_nose  = True

                    print(np.argmax(result[0]))
                   
                
                
                sub_image = resultImage[t_y:t_y+t_h, t_x:t_x+t_w]
                # print(sub_image.shape)
                #print(layer.shape)
               # cv2.imwrite('image.jpg', sub_image )
                alpha = 1
                beta = 1- alpha
                res = cv2.addWeighted(sub_image, alpha, layer, beta , 1)

                
                
                if res.ndim == 3:
                    res = cv2.resize(res,(int(96),int(96)))
                    if show_mouth == True:
                        index_mounth = 25
                        index_eye_left = 1
                        size = 20
                        size_eye = 17

                        aux = res[int(keypoint[index_mounth]):int(keypoint[index_mounth])+size, int(keypoint[index_mounth -1]):int(keypoint[index_mounth -1])+size ]
                        tongue = cv2.resize(tongue,(int(aux.shape[1]),int(aux.shape[0])))

                       
                        
                        res[int(keypoint[index_mounth]):int(keypoint[index_mounth])+size, int(keypoint[index_mounth -1]):int(keypoint[index_mounth -1])+size ] = tongue

                        eyes_left = cv2.resize(eyes_left,(int(size_eye),int(size_eye)))

                        res[int(keypoint[index_eye_left]):int(keypoint[index_eye_left])+size_eye, int(keypoint[index_eye_left -1]):int(keypoint[index_eye_left -1])+size_eye] = eyes_left
                    elif show_nose == True:
                        index_nose = 20
                        index_mounth = 25
                        #index_eye_right = 3
                        size = 25
                        size_mouth = 25

                        #print("add nose")
                        #aux = res[int(keypoint[index_nose]):int(keypoint[index_nose])+size, int(keypoint[index_nose -1]):int(keypoint[index_nose -1])+size ] 

                        #nose = cv2.resize(nose,(int(aux.shape[1]),int(aux.shape[0])))
                        #res[int(keypoint[index_nose]):int(keypoint[index_nose])+size, int(keypoint[index_nose -1]):int(keypoint[index_nose -1])+size ] = nose

                        aux = res[int(keypoint[index_mounth]):int(keypoint[index_mounth])+size, int(keypoint[index_mounth -1]):int(keypoint[index_mounth -1])+size ]
                        mouth = cv2.resize(mouth,(int(aux.shape[1]),int(aux.shape[0])))

                        print("add mouth"   + str(mouth.shape))

                        res[int(keypoint[index_mounth]):int(keypoint[index_mounth])+size_mouth, int(keypoint[index_mounth -1]):int(keypoint[index_mounth -1])+size_mouth ] = mouth
                        #res[int(keypoint[index_eye_right]):int(keypoint[index_eye_right])+size_eye, int(keypoint[index_eye_right -1]):int(keypoint[index_eye_right -1])+size_eye] = rigth_left
                    elif show_point ==  True:
                        for j in range(1, len(keypoint),2):
                            cv2.circle(res, (keypoint[j -1],keypoint[j]) , 1, (0,255,0), 1)
                
                res = cv2.resize(res,(int(t_w),int(t_h)))
                resultImage[t_y:t_y+t_h, t_x:t_x+t_w] = cv2.resize(res, (t_w,t_h), interpolation = cv2.INTER_CUBIC)
            except Exception as inst:
                print(inst)
                pass


            #
            #image_new = cv2.addWeighted(resultImage, alpha, image, 1 - alpha, 0)



        else:

            cv2.putText(resultImage, "Detecting..." , 
                            (int(t_x + t_w/2), int(t_y)), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 130, 75), 2)

    # Our operations on the frame come here
    #rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Display the resulting frame
    baseImage = resultImage
    cv2.imshow('Selfie',baseImage)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

       


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()