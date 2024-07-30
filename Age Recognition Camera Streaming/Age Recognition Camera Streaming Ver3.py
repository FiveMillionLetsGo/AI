import cv2
import numpy as np
from scipy.signal import convolve2d

# Cascade 분류기 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def estimate_age(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    # 텍스처 분석을 위한 Gabor 필터
    def gabor_filter(ksize=31):
        gabor_kern = cv2.getGaborKernel((ksize, ksize), 4.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        return gabor_kern

    # Gabor 필터 적용
    gabor = gabor_filter()
    filtered = convolve2d(gray, gabor, mode='same', boundary='symm')
    
    # 피부 텍스처 분석
    texture_feature = np.std(filtered)
    
    # 주름 검출
    edges = cv2.Canny(gray, 50, 150)
    wrinkle_density = np.sum(edges) / (face.shape[0] * face.shape[1])
    
    # 눈 주변 영역 분석
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    eye_area_ratio = sum([w*h for (x,y,w,h) in eyes]) / (face.shape[0] * face.shape[1]) if len(eyes) > 0 else 0
    
    # 피부 톤 분석
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    skin_tone = np.mean(hsv[:,:,1])  # 채도 평균
    
    # 나이 추정 로직
    if wrinkle_density > 0.1 and texture_feature > 20:
        return "60+"
    elif wrinkle_density > 0.08 and texture_feature > 18:
        return "50-60"
    elif wrinkle_density > 0.06 and texture_feature > 15:
        return "40-50"
    elif wrinkle_density > 0.04 and texture_feature > 12:
        return "30-40"
    elif eye_area_ratio > 0.05 and skin_tone > 50:
        return "20-30"
    else:
        return "0-20"

# 메인 루프
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