import cv2

def main():
    cap = cv2.VideoCapture(0)  # 0: 기본 카메라 / 다른 카메라 필요시 번호 확인 필요

    if not cap.isOpened():
        print("카메라 작동 불가")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("비디오 작동 불가")
            break

        cv2.imshow('Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()