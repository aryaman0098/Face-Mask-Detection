import numpy as np
from keras.models import load_model
import cv2
#import matplotlib.pyplot as plt


model1 = load_model("Model1.h5")#Loading the first CNN model
model2 = load_model("Model2.h5")#Loading the second CNN model

face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")#Loading the harcascade classfier for face detecction

cam = cv2.VideoCapture(0)

labels = ["No Mask", "Mask"]

while True:
    ret, frame = cam.read()

    frame1 = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)#Capturing the current frame and converting it into RGB from BGR

    faces = face.detectMultiScale(frame, 1.3,5)#Detecting face
    
    for x, y, w, h in faces:
        
        face_img = frame[y:y+w, x:x+w]
        smallFrame = cv2.resize(frame1, (150, 150))#Resizing the image so that it can be fed into the CNN
        smallFrameNormalised = smallFrame / 255.0
        input = np.reshape(smallFrameNormalised, (1, 150, 150, 3))

        ensemble1 = model1.predict(input)#Predicting the outcome using the pretrained CNN model1
        ensemble2 = model2.predict(input)#Predicting the outcome using the pretrained CNN model2

        prediction = [(ensemble1[0][0] + ensemble2[0][0])/2.0, (ensemble1[0][1] + ensemble2[0][1])/2.0] #Taking the mean of both the predictions

        if prediction[0] < prediction[1]:
            prob = str(round(prediction[1] * 100, 2))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)#Bounding box to display the outcome
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (0, 255, 0), -1)
            cv2.putText(frame, labels[1] + " " +prob, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, .90, (255, 255, 255), 2)
        
        elif prediction[0] > prediction[1]:
            prob = str(round(prediction[0] * 100, 2))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)#Bounding box to display the outcome
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (0, 0, 255), -1)
            cv2.putText(frame, labels[0] + " " +prob, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, .90, (255, 255, 255), 2)
        
        print(prediction)

    # plt.imshow(np.reshape(input, (150, 150, 3)))
    # plt.show()
    # break

    cv2.imshow("Video", frame)

    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
