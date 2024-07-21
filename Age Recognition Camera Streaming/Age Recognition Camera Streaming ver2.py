# 라이브러리 설치
# pip install opencv-python numpy
import cv2
import numpy as np

# face_cascade : 얼굴 검출 Cascade객체
# eye_cascade : 눈 검출 Cascade객체
# cv2.CascadeClassfier() : OpenCv Cascade Classfier 생성 함수 : CascadeClassfier : 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def estimate_age(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    avg_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    edge_intensity = np.mean(cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5))
    
    eyes = eye_cascade.detectMultiScale(gray)
    eye_count = len(eyes)
    
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    wrinkle_count = len(contours)
    
    if wrinkle_count > 100 and edge_intensity > 35:
        return "60+"
    elif wrinkle_count > 70 and edge_intensity > 30:
        return "50-60"
    elif wrinkle_count > 50 and edge_intensity > 25:
        return "40-50"
    elif wrinkle_count > 30 and edge_intensity > 20:
        return "30-40"
    elif avg_intensity > 130 and std_intensity < 50:
        return "20-30"
    else:
        return "0-20"

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        age_range = estimate_age(face)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Age: {age_range}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Age Range Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()