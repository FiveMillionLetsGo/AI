# 주석 없는 version

import numpy as np
import speech_recognition as sr
from scipy.signal import butter, lfilter

menu_list = ["카푸치노", "아메리카노", "에스프레소", "바닐라라떼", "딸기라떼"]

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def noise_reduction(audio_data):
    audio_array = np.frombuffer(audio_data.get_raw_data(), np.int16)
    lowcut = 300.0
    highcut = 3400.0
    fs = audio_data.sample_rate
    audio_filtered = butter_bandpass_filter(audio_array, lowcut, highcut, fs, order=6)
    return sr.AudioData(audio_filtered.astype(np.int16).tobytes(), audio_data.sample_rate, audio_data.sample_width)

def recognize_speech_from_mic(attempts=3):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    for attempt in range(attempts):
        print(f"시도 횟수 : {attempt + 1}/{attempts}")
        
        with mic as source:
            print("소음 조절")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("주문")
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            except sr.WaitTimeoutError:
                print("시간 초과. 다시 시도해주세요.")
                continue
        
        try:
            denoised_audio = noise_reduction(audio)
            print("음성을 인식하는 중.")
            response = recognizer.recognize_google(denoised_audio, language="ko-KR")
            print(f"인식된 음성: {response}")
            return response.lower()
        
        except sr.RequestError as event:
            print(f"API를 사용할 수 없거나 응답이 없습니다. 오류: {event}")
            return None
        except sr.UnknownValueError:
            print("음성 인식 실패 : 재시도")
    
    return None

def match_menu_order(recognized_text):
    for menu_item in menu_list:
        if menu_item in recognized_text:
            return menu_item
    return "일치하는 메뉴가 없습니다."

if __name__ == "__main__":
    print("음성 인식")
    recognized_text = recognize_speech_from_mic()
    
    if recognized_text:
        selected_menu = match_menu_order(recognized_text)
        print(f"선택된 메뉴 : {selected_menu}")
    
    else:
        print("음성을 인식 실패 / 음성 입력이 없음")