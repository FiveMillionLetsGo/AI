import cv2

def main():
    cap = cv2.VideoCapture(0)  # 0은 기본 카메라를 의미합니다. 다른 카메라를 사용할 경우 숫자를 변경할 수 있습니다.

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("비디오를 읽을 수 없습니다.")
            break

        # 여기에 프레임 처리 로직을 추가할 수 있습니다 (예: 화면에 출력)
        cv2.imshow('Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()