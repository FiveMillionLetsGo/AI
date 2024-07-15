import speech_recognition as sr

def test_microphone():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("음성 입력")
        audio = recognizer.listen(source, timeout=5)
    
    try:
        text = recognizer.recognize_google(audio, language="ko-KR")
        print(f"인식된 텍스트: {text}")
    except sr.UnknownValueError:
        print("음성 인식 실패")
    except sr.RequestError as e:
        print(f"Google 음성 인식 서비스 오류 발생 {e}")

if __name__ == "__main__":
    test_microphone()