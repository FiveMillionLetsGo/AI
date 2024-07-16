# 주석으로 설명한 version 

# 수치 계산 라이브러리
import numpy as np
# 음성 인식 라이브러리
import speech_recognition as sr
# 신호 처리 라이브러리
from scipy.signal import butter, lfilter

# 메뉴 리스트
menu_list = ["카푸치노", "아메리카노", "에스프레소", "바닐라라떼", "딸기라떼"]

# 주파수 대역대 설정 : 밴드패스 필터 설계 : Butterworth 밴드패스 필터
# 원리 : 특정 주파수 대역만 통과시키고 나머지는 제거

# 통과시킬 주파수 : lowcut ~ highcut
# lowcut : 통과시킬 주파수의 하한값
# highcut : 통과시킬 주파수의 상한값

# fs : 샘플링 주파수(초당 샘플 수)
# order : 필터의 차수(기본 5)

def butter_bandpass(lowcut, highcut, fs, order=5):
    # 나이퀴스트 주파수 계산 : (나이퀴스트 주파수 = 샘플링 주파수 / 2)
    # 나이퀴스트 주파수 : 디지털 신호 표현 최대 주파수
    nyq = 0.5 * fs
    
    # 버터워스 필터 (0~1사이의 정규화된 주파수 사용)
    # 주파수 정규화 : lowcut ~ highcut 주파수 대역대
    low = lowcut / nyq
    high = highcut / nyq
    
    # butter 함수 = scipy.signal.butter함수
    # order : 필터 차수(높을 수록 성능 개선,계산복잡도 증가)
    #                (낮을 수록 성능 하향,계산복잡도 감소)
    # [low,high] : 정규화된 차단 주파수 리스트
    # btype : 'band' : 밴드패스 필터 지정 
    b, a = butter(order, [low, high], btype='band')
    
    # 계산된 필터 계수 반환
    # b : 필터의 분자 계수 : 입력 신호의 가중치
    # a : 필터의 분모 계수 : 이전 출력 값의 가중치
    return b, a

# 실제 입력 신호에 밴드패스 필터 적용
# data : 필터링할 입력 신호 데이터
# lowcut : 통과시킬 주파수의 하한값
# highcut : 통과시킬 주파수의 상한값
# fs : 샘플링 주파수 : 초당 샘플 수
# order : 필터 차수
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    # butter_bandpass함수를 호출하여 필터 계수 b와 a를 얻음
    # 계수들(b와a)은 지정된 주파수 범위를 통과시키는 밴드패스 필터 정의
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # lfilter :  실제 필터링 수행(관련 내용 : scipy.signal.lfilter 함수 사용
    # b,a : 계수들
    # data : 필터링할 입력 신호
    # data에 필터를 적용하여 필터링된 신호 y생성
    y = lfilter(b, a, data)
    # 필터링된 신호 y 반환
    return y


# 오디오 데이터 소음 감소 기능
def noise_reduction(audio_data):
    # ----- 소음 감소 진행 -----
    # 소음 감소
    # audio_data.get_raw_data() : 원본 데이터를 가져 옵니다.(입력시에 받은 전체 내용)
    # np.frombuffer : 해당 원시 데이터를 16비트 정수 numpy 배열로 반환
    # np.frombuffer를 거친 numpy배열이 audio_array에 들어가게됩니다.
    audio_array = np.frombuffer(audio_data.get_raw_data(), np.int16)
    
    # 일반적인 인간 음성 주파수 : 300Hz(lowcut) ~ 3400Hz(highcut)
    # 밴드패스 필터 하한값
    lowcut = 300.0
    # 밴드 패스 필터 상한값
    highcut = 3400.0
    
    # 오디오 샘플링 주파수
    fs = audio_data.sample_rate
    
    # audio_filtered : butter_bandpass_filter로 오디오 데이터 필터링
    audio_filtered = butter_bandpass_filter(audio_array, lowcut, highcut, fs, order=6)
    
    # ----- 소음 감소 완료 -------
    
    # 필터링 audio를 speech_recognition.AudioData 객체로 변환 후 반환
    # audio_filtered(np.int16).tobytes() : 필터링된 데이터를 16비트 정수 변환 후 바이트 형식으로 변환 
    return sr.AudioData(audio_filtered.astype(np.int16).tobytes(), audio_data.sample_rate, audio_data.sample_width)

# recognize_speech_from_mic : 마이크 인식
# attempys : 인식하지 못하면 시도할 횟수
def recognize_speech_from_mic(attempts=3):
    # recognizer : sr.Recognizer함수로 음성 인식기 객체 생성
    recognizer = sr.Recognizer()
    # mic : sr.Microphone함수로 마이크 객체 생성
    mic = sr.Microphone()
    
    # attemps 숫자만큼 음성인식 시도
    for attempt in range(attempts):
        # 시도 횟수 출력
        print(f"시도 횟수 : {attempt + 1}/{attempts}")
        
        # 마이크를 음성 소스로 사용
        with mic as source:
            print("소음 조절")
            # 주변 소음 조정 : 1초동안 주변 소음 분석 후 음성 인식기 조절 : adjust_for_ambient_noise 함수로 소음 조절
            recognizer.adjust_for_ambient_noise(source, duration=1)
            # 주문을 위한 내용
            print("주문")
            # audio : 음성인식(사용자의 말)
            # recognizer.listen 함수로 인식을 받음
            # timeout : 음성 인식까지 기다리는 시간
            # phrase_time_limit : 음성인식까지 기다리는 시간
            # timeout동안 음성인식을 기다리며 pharse_time_limit이상 음성이 지속되면 자동으로 끊음
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            #sr.WaitTimeoutError : 시간 초과시 다음으로 넘어감
            except sr.WaitTimeoutError:
                print("시간 초과. 다시 시도해주세요.")
                continue
        
        try:
            # denoised_audio : noise_reduction에 입력받은 오디오를 매개변수로 넣어 소음을 줄임
            denoised_audio = noise_reduction(audio)
            
            # 음성 인식을 거쳤다는 것을 알 수 있도록 확인용 print구문
            print("음성을 인식하는 중.")
            
            # response : recognizer.recognize_google 구글에서 지원하는 API이용해서 denoised_audio : 음성 제거된 음성파일을 넣어서 한국어로 인식시킴
            # 요약 : 소음 제거된 입력된 음성을 한국어로 인식하는 코드
            response = recognizer.recognize_google(denoised_audio, language="ko-KR")
            
            #인식이 어떻게 되었는지 확인
            print(f"인식된 음성: {response}")
            
            # 인식된 텍스트를 소문자로 변환하여 반환 : 대소문자 구분 없이 메뉴 매칭을 위한 코드
            return response.lower()
        
        # API 사용시 오류가 생기면 해당 API 확인용 코드
        except sr.RequestError as event:
            print(f"API를 사용할 수 없거나 응답이 없습니다. 오류: {event}")
            # 오류 시 인식 실패를 반환
            return None
        # 음성을 텍스트로 변환 할 수 없는 경우 처리
        except sr.UnknownValueError:
            print("음성 인식 실패 : 재시도")
    
    # 실패시 반환 음성인식 실패 반환(시도 횟수가 끝나면 다시 실패 반환)
    return None

# 인식된 음성과 menu가 일치하는지 확인하는 함수
# recognized_text : 인식된 텍스트 : 소음제거 / API 과정까지 거친 후 인식된 음성
def match_menu_order(recognized_text):
    # menu_list 순회하며 menu_item 를 찾음 : 처음부터 돌면서 아래 recognized_text를 찾는 내용
    for menu_item in menu_list:
        # 만약 menu_item과 recognized_text가 일치하면 menu_item 반환
        if menu_item in recognized_text:
            return menu_item
    # 해당 메뉴가 없다면 메뉴 없다라는 내용을 반환함
    return "일치하는 메뉴가 없습니다."

# 함수 main
if __name__ == "__main__":
    # 음성인식 시작을 알림
    print("음성 인식")
    # recognized_text : recognize_speech_from_mic 함수 호출하여 음성 인식 : 음성을 텍스트로 변환된 결과 반환
    recognized_text = recognize_speech_from_mic()
    
    # 만약 인식된 음성(recognized_text)이 있다면 = null이 아니므로 -> true / Truethy,Falsy
    if recognized_text:
        # 인식된 텍스트에서 메뉴를 찾는다.
        # selected_menus :고른 항목
        # match_menu_order : recognized_text 인식된 텍스트를 메뉴 일치하는지 확인
        selected_menu = match_menu_order(recognized_text)
        # 선택된 메뉴 출력
        print(f"선택된 메뉴 : {selected_menu}")
    
    # 만약 인식 안됬다면 음성인식 실패 코드 출력
    else:
        print("음성을 인식 실패 / 음성 입력이 없음")