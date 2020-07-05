import numpy as np
from keras.models import load_model
import cv2


model = load_model("Model.h5")

face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cam = cv2.VideoCapture(0)

labels = ["No Mask", "Mask"]

while True:
    ret, frame = cam.read()

    
    faces = face.detectMultiScale(frame, 1.3,5)
    
    for x, y, w, h in faces:
        
        face_img = frame[y:y+w, x:x+w]
        smallFrame = cv2.resize(face_img, (150, 150))
        smallFrameNormalised = smallFrame / 255.0
        input = np.reshape(smallFrameNormalised, (1, 150, 150, 3))

        prediction = model.predict(input)

        label = np.argmax(prediction, axis = 1)[0]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 122), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (200, 0, 122), -1)
        
        if prediction[0][0] < prediction[0][1]:
            cv2.putText(frame, labels[1], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        elif prediction[0][0] > prediction[0][1]:
            cv2.putText(frame, labels[0], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        print(prediction)

    cv2.imshow("Video", frame)

    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
