import numpy as np
from keras.models import load_model
import cv2


model = load_model("Model.h5")#Loading the CNN model


face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")#Loading the harcascade classfier for face detecction

cam = cv2.VideoCapture(0)

labels = ["No Mask", "Mask"]

#fourcc = cv2.VideoWriter_fourcc(*'XVID') 
#out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))  #Uncomment these two lines if you want to save the recording of the web cam

while True:
    ret, frame = cam.read()

    frame1 = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)#Capturing the current frame and converting it into RGB from BGR

    faces = face.detectMultiScale(frame, 1.3,5)#Detecting face
    
    for x, y, w, h in faces:
        
        face_img = frame[y:y+w, x:x+w]
        smallFrame = cv2.resize(frame1, (150, 150))#Resizing the image so that it can be fed into the CNN
        smallFrameNormalised = smallFrame / 255.0
        input = np.reshape(smallFrameNormalised, (1, 150, 150, 3))


        prediction = model.predict(input)

        if prediction[0][0] < prediction[0][1]:
            prob = str(round(prediction[0][1] * 100, 2))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)#Bounding box to display the prediction
            cv2.rectangle(frame, (x, y - 35), (x + w, y), (0, 255, 0), -1)
            cv2.putText(frame, labels[1] + " " +prob, (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, .70, (255, 255, 255), 2)
        
        elif prediction[0][0] > prediction[0][1]:
            prob = str(round(prediction[0][0] * 100, 2))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)#Bounding box to display the prediction
            cv2.rectangle(frame, (x, y - 35), (x + w, y), (0, 0, 255), -1)
            cv2.putText(frame, labels[0] + " " +prob, (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, .70, (255, 255, 255), 2)
        
        #out.write(frame) #Uncomment this line if you want to save the recording of the web cam

    cv2.imshow("Video", frame)

    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
