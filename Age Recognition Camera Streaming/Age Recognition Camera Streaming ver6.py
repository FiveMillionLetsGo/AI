import cv2
import numpy as np
from deepface import DeepFace
import time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_interval = 15
frame_count = 0
last_age = None
smooth_factor = 0.3
age_buffer = []
buffer_size = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % frame_interval == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            
            try:
                resized_face = cv2.resize(face_img, (224, 224))
                resized_face = cv2.fastNlMeansDenoisingColored(resized_face, None, 10, 10, 7, 21)
                
                analysis = DeepFace.analyze(resized_face, actions=['age'], enforce_detection=False)
                current_age = analysis[0]['age']
                
                age_buffer.append(current_age)
                if len(age_buffer) > buffer_size:
                    age_buffer.pop(0)
                
                smoothed_age = np.mean(age_buffer)
                
                if last_age is None:
                    last_age = smoothed_age
                else:
                    last_age = last_age * (1 - smooth_factor) + smoothed_age * smooth_factor
                
                age = int(last_age)
                
                print(f"Detected Age: {age}")
            except Exception as e:
                print(f"DeepFace analysis error: {e}")
                age = "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Age: {age}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Age Prediction', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
