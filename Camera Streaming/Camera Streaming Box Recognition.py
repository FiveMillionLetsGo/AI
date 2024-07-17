import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

    if not cap.isOpened():
        print("카메라 작동 불가")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("비디오 작동 불가")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        profiles = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            x, y, w, h = x*2, y*2, w*2, h*2
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            roi_gray = gray[y//2:(y+h)//2, x//2:(x+w)//2]
            roi_color = frame[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex*2, ey*2), (ex*2+ew*2, ey*2+eh*2), (255, 0, 0), 2)

        for (x, y, w, h) in profiles:
            x, y, w, h = x*2, y*2, w*2, h*2
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

        additional_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
        for (x, y, w, h) in additional_faces:
            x, y, w, h = x*2, y*2, w*2, h*2
            if not any(abs(x-fx) < w/2 and abs(y-fy) < h/2 for (fx, fy, fw, fh) in faces):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        cv2.imshow('Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()