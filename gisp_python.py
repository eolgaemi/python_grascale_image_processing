# import 부
#### tkinter ####
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import *
from tkinter.simpledialog import *

## etc ##
import os.path
import math

# 전역변수부
window = None
canvas = None
paper = None
in_height, in_width, out_height, out_width = [0] * 4
inImage = outImage = []

# 함수부
## 공통 함수부

### 이미지 열기
def open_image():
    # 전역변수
    global window, canvas, paper
    global inImage, in_height, in_width

    # 0. 영상의 위치 저장
    image_path = askopenfilename(parent=window, filetypes=(("RAW파일", "*.raw"), ("모든파일", "*.*")))

    # 1. 영상 크기 결정
    fsize = os.path.getsize(image_path)
    in_height = in_width = int(math.sqrt(fsize))

    # 2. 메모리 확보
    inImage = [[0 for _ in range(in_width)] for _ in range(in_height)]

    # 3. 영상 바이너리로 읽기 포인터 생성
    rfp = open(image_path, "rb")
    for h in range(in_height):
        for w in range(in_width):
            # ord = 아스키 코드,유니 코드 읽어주기 <-> 반대는 chr
            inImage[h][w] = ord(rfp.read(1))

    # 4. 읽기 포인터 제거
    rfp.close()

    # 5. 읽은 영상 display
    display_image(inImage)

    # 6. 바로 복사본 만들기
    equal_image()


### 이미지 화면에 띄우기
def display_image(image):
    # 전역변수
    global window, paper, canvas

    # 높이, 너비 가져오기
    height = len(image)
    width = len(image[0])

    # 기존에 이미지를 떼어내기
    if (canvas != None):
        canvas.destroy()

    # 새로운 window, canvas, paper 만들기
    # 윈도우 창 크기 조절
    window.geometry(str(width) + "x" + str(height))

    # 윈도우에 새로운 캔버스 만들기
    canvas = Canvas(window, height=height, width=width, bg="yellow")

    # 점을 찍을 paper 새로 만들기
    paper = PhotoImage(height=height, width=width)

    # 캔버스에 paper를 걸기
    canvas.create_image((width // 2, height // 2), image=paper, state="normal")

    '''
    ### BAD WAY --> 한 점씩 가져놓고 16진수로 변환하고 한 점 찍기
    for h in range(height):
        for w in range(width):
            r = g = b = image[h][w]
            paper.put("#%02x%02x%02x" % (r, g, b), (w, h))
    '''

    # 이미지를 빠르게 가져오는 법(더블 버퍼링) <-- GOOD WAY
    # 메모리에 올려서 한꺼번에 16진수변환 -> 그대로 옮기기

    # 전체에 대한 16진수 문자열
    rgbString = ""

    # {a1 b7 cf} {c5 ba fe} 형식으로 문자열 만들기
    for h in range(height):
        one_string = ""
        for w in range(width):
            # 한 줄에 대한 16진수 문자열
            r = g = b = image[h][w]
            one_string += "#%02x%02x%02x " % (r, g, b)
        rgbString += "{" + one_string + "} "

    # 문자열을 기반으로 점 찍기
    paper.put(rgbString)

    # 배치하기
    canvas.pack()


### 이미지 저장하기
def save_image():
    # 전역변수
    global window
    global outImage

    # 출력 이미지가 없다면 바로 동작 X
    if (outImage == None or len(outImage) == 0):
        return

    # 출력 이미지가 있다면 저장 위치 파일 포인터 만들기
    wfp = asksaveasfile(parent=window, mode="wb", defaultextension="*.raw",
                        filetypes=(("RAW파일", "*.raw"), ("모든파일", "*.*")))

    # 출력이미지 높이와 너비
    height = len(outImage)
    width = len(outImage[0])

    # 포맷: B와 이미지를 구조체로 만들어서 write
    import struct
    for h in range(height):
        for w in range(width):
            wfp.write(struct.pack("B", outImage[h][w]))
    # 파일 포인터 닫기
    wfp.close()
    # 저장완료 메세지 박스 보여주기
    messagebox.showinfo("성공", wfp.name + "저장완료")

def exit_app():
    global window
    if messagebox.askokcancel("종료","종료하시겠습니까?"):
        window.destroy()

### 원본 이미지 그대로 출력 이미지로 복사
def equal_image():
    # 전역 변수
    global inImage, in_width,in_height
    global outImage, out_width, out_height

    # 입력 이미지의 높이와 너비 복사하기
    out_height = len(inImage)
    out_width = len(inImage[0])

    # 출력 이미지 0으로 초기화
    outImage = [[0 for _ in range(out_width)] for _ in range(out_height)]

    # 입력 이미지 --> 출력 이미지로 복사
    for h in range(out_height):
        for w in range(out_width):
            outImage[h][w] = inImage[h][w]

    # 복사 하자마자 화면에 출력하기
    display_image(outImage)


## 화소점 처리 함수부
### 밝기 조절
def trans_brightness_pm():
    # 전역 변수
    global outImage

    # 밝기 값 입력
    iVal = askinteger("밝기 조절 강도 입력", "-255~255 입력", maxvalue=255, minvalue=-255)

    # 입력 이미지의 높이와 너비 복사하기
    out_height = len(outImage)
    out_width = len(outImage[0])

    # 입력 이미지 +- 밝기 계산
    for h in range(out_height):
        for w in range(out_width):
            temp = outImage[h][w] + iVal
            if (temp > 255):
                outImage[h][w] = 255
            elif (temp < 0):
                outImage[h][w] = 0
            else:
                outImage[h][w] = temp

    # 화면에 출력
    display_image(outImage)

### 밝기 조절
def trans_brightness_mul():
    # 전역 변수
    global outImage

    # 밝기 값 입력
    iVal = askinteger("밝기 조절(*) 강도 입력", "0~255 입력", maxvalue=255, minvalue=0)

    # 입력 이미지의 높이와 너비 복사하기
    out_height = len(outImage)
    out_width = len(outImage[0])

    # 입력 이미지 */ 밝기 계산
    for h in range(out_height):
        for w in range(out_width):
            temp = outImage[h][w] * iVal
            if (temp > 255):
                outImage[h][w] = 255
            elif (temp < 0):
                outImage[h][w] = 0
            else:
                outImage[h][w] = temp

    # 화면에 출력
    display_image(outImage)

### 밝기 조절
def trans_brightness_div():
    # 전역 변수
    global outImage

    # 밝기 값 입력
    iVal = askfloat("밝기 조절(/) 강도 입력", "0.0~255.0 입력", maxvalue=255.0, minvalue=0.0)

    # 입력 이미지의 높이와 너비 복사하기
    out_height = len(outImage)
    out_width = len(outImage[0])

    # 입력 이미지 */ 밝기 계산
    for h in range(out_height):
        for w in range(out_width):
            temp = outImage[h][w] / iVal
            if (temp > 255):
                outImage[h][w] = 255
            elif (temp < 0):
                outImage[h][w] = 0
            else:
                outImage[h][w] = int(temp)

    # 화면에 출력
    display_image(outImage)


### 감마 보정
def trans_gamma():
    # 전역 변수
    global outImage

    # 입력 이미지의 높이와 너비 복사하기
    out_height = len(outImage)
    out_width = len(outImage[0])

    # 감마 보정 강도 입력
    fVal = askfloat("감마 보정 강도 입력", "0.0~2.0 입력", maxvalue=2.0, minvalue=0.0)

    # gamma processing.....
    for h in range(out_height):
        for w in range(out_width):
            outImage[h][w] = int(255 * (outImage[h][w] / 255) ** (1.0 / fVal))

    display_image(outImage)


### 흑백 처리
def trans_bow():
    # 전역 변수
    global outImage

    # 입력 이미지의 높이와 너비 복사하기
    out_height = len(outImage)
    out_width = len(outImage[0])

    # 흑백 처리 기준값 설정
    iVal = askinteger("흑백 처리 기준값 입력", "0~255 입력", maxvalue=255, minvalue=0)

    # 기준값으로 흑백 나누기
    for h in range(out_height):
        for w in range(out_width):
            if outImage[h][w] < iVal:
                outImage[h][w] = 0
            else:
                outImage[h][w] = 255

    # 화면에 출력
    display_image(outImage)


###반전
def trans_reverse():
    # 전역 변수
    global outImage

    # 입력 이미지의 높이와 너비 복사하기
    out_height = len(outImage)
    out_width = len(outImage[0])

    # 반전시키기
    for h in range(out_height):
        for w in range(out_width):
            outImage[h][w] = 255 - outImage[h][w]

    # 화면에 출력
    display_image(outImage)


###AND
def trans_and():
    # 전역 변수
    global outImage

    # 입력 이미지의 높이와 너비 복사하기
    out_height = len(outImage)
    out_width = len(outImage[0])

    # AND 기준 값 설정
    iVal = askinteger("AND 처리 기준값 입력", "0~255 입력", maxvalue=255, minvalue=0)

    # AND 처리
    for h in range(out_height):
        for w in range(out_width):
            outImage[h][w] = outImage[h][w] & iVal

    # 화면에 출력
    display_image(outImage)


###OR
def trans_or():
    # 전역 변수
    global outImage

    # 입력 이미지의 높이와 너비 복사하기
    out_height = len(outImage)
    out_width = len(outImage[0])

    # OR 기준 값 설정
    iVal = askinteger("OR 처리 기준값 입력", "0~255 입력", maxvalue=255, minvalue=0)

    # OR 처리
    for h in range(out_height):
        for w in range(out_width):
            outImage[h][w] = outImage[h][w] | iVal

    # 화면에 출력
    display_image(outImage)


###XOR
def trans_xor():
    # 전역 변수
    global outImage

    # 입력 이미지의 높이와 너비 복사하기
    out_height = len(outImage)
    out_width = len(outImage[0])

    # XOR 기준 값 설정
    iVal = askinteger("XOR 처리 기준값 입력", "0~255 입력", maxvalue=255, minvalue=0)

    # XOR 처리
    for h in range(out_height):
        for w in range(out_width):
            outImage[h][w] = outImage[h][w] ^ iVal

    # 화면에 출력
    display_image(outImage)


### 파라볼라 CAP
def trans_para_cap():
    # 전역 변수
    global outImage

    # 입력 이미지의 높이와 너비 복사하기
    out_height = len(outImage)
    out_width = len(outImage[0])

    # para_cap 처리
    for h in range(out_height):
        for w in range(out_width):
            temp = int(255 * ((outImage[h][w] / 127 - 1) ** 2))
            outImage[h][w] = 255 if temp > 255 else 0 if temp < 0 else int(temp)

    # 화면에 출력
    display_image(outImage)


### 파라볼라 CUP
def trans_para_cup():
    # 전역 변수
    global outImage

    # 입력 이미지의 높이와 너비 복사하기
    out_height = len(outImage)
    out_width = len(outImage[0])

    # para_cup 처리
    for h in range(out_height):
        for w in range(out_width):
            temp = int(-255 * (outImage[h][w] / 127 - 1) ** 2 + 255)
            outImage[h][w] = 255 if temp > 255 else 0 if temp < 0 else int(temp)

    # 화면에 출력
    display_image(outImage)


### 히스토그램 스트레치
def trans_hist_stretch():
    # 전역 변수
    global outImage

    # 입력 이미지의 높이와 너비 복사하기
    out_height = len(outImage)
    out_width = len(outImage[0])

    # 최대 명도
    highest = 0
    # 최소 명도
    lowest = 256

    # 최대,최소 명도 구하기
    for h in range(out_height):
        for w in range(out_width):
            if highest < outImage[h][w]:
                highest = outImage[h][w]
            if lowest > outImage[h][w]:
                lowest = outImage[h][w]

    # 최대,최소 명도를 이용해서 정규화
    for h in range(out_height):
        for w in range(out_width):
            outImage[h][w] = int((outImage[h][w] - lowest) / (highest - lowest) * 255.0)

    # 화면에 출력
    display_image(outImage)


### 히스토그램 엔드 인 스트레치
def trans_hist_endin_stretch():
    # 전역 변수
    global outImage

    # 입력 이미지의 높이와 너비 복사하기
    out_height = len(outImage)
    out_width = len(outImage[0])

    # 최대 명도
    highest = 0
    # 최소 명도
    lowest = 256

    # 최대,최소 명도 구하기
    for h in range(out_height):
        for w in range(out_width):
            if highest < outImage[h][w]:
                highest = outImage[h][w]
            if lowest > outImage[h][w]:
                lowest = outImage[h][w]

    # input end-in value
    iVal = askinteger("END-IN 처리 기준값 입력", "0~255 입력", maxvalue=255, minvalue=0)
    # 적절히 endin 조절
    highest -= iVal
    lowest += iVal
    # 최대,최소 명도를 이용해서 정규화
    for h in range(out_height):
        for w in range(out_width):
            temp = int((outImage[h][w] - lowest) / (highest - lowest) * 255.0)
            outImage[h][w] = 255 if temp > 255 else 0 if temp < 0 else int(temp)

    # 화면에 출력
    display_image(outImage)


### 히스토그램 평활화
def trans_hist_equal():
    # 전역 변수
    global outImage

    # 입력 이미지의 높이와 너비 복사하기
    out_height = len(outImage)
    out_width = len(outImage[0])

    # 히스토그램(도수 분포표) 초기화
    histo_gram = [0 for _ in range(out_height * out_width)]

    # 누적 히스토그램 (누적 도수 분포표) 초기화
    sum_histo_gram = [0 for _ in range(out_height * out_width)]

    # 정규화 히스토그램 초기화
    norm_histo_gram = [0 for _ in range(out_height * out_width)]

    # 총 픽셀 수
    n_pixels = out_height * out_width

    # 최대 명도
    highest = 0

    # 최대,최소 명도 구하기
    for h in range(out_height):
        for w in range(out_width):
            if highest < outImage[h][w]:
                highest = outImage[h][w]

    # 히스토그램(도수 분포표) 만들기
    for h in range(out_height):
        for w in range(out_width):
            histo_gram[outImage[h][w]] += 1
    # 0번째 값만 복사
    sum_histo_gram[0] = histo_gram[0]
    # 누적 히스토그램 (누적 도수 분포표) 만들기
    for i in range(1, len(sum_histo_gram) - 1, 1):
        sum_histo_gram[i] = sum_histo_gram[i - 1] + histo_gram[i]
    # 정규화 히스토그램 만들기
    for i in range(len(sum_histo_gram)):
        norm_histo_gram[i] = int(sum_histo_gram[i]*(1.0/n_pixels)*highest)


    # 정규화 한 결과를 outImage에 저장
    for h in range(out_height):
        for w in range(out_width):
            outImage[h][w] = norm_histo_gram[outImage[h][w]]

    # 화면에 출력
    display_image(outImage)


## 영상 통계 함수부
### 평균
def get_average(image):

    # 입력 이미지의 높이와 너비 복사하기
    height = len(image)
    width = len(image[0])

    # 모든 이미지 화소값의 총합구하기
    sum = 0
    for h in range(height):
        for w in range(width):
            sum += image[h][w]

    # 평균 리턴하기
    return sum / (height * width)


### 중앙값
def get_median(image):

    # 입력 이미지의 높이와 너비 복사하기
    height = len(image)
    width = len(image[0])

    # 인덱스로 중앙값을 찾기 위해 평평하게 만들기
    image_flat = [0 for _ in range(height * width)]

    for h in range(height):
        for w in range(width):
            image_flat[h * w + w] = image[h][w]

    # 정렬하기
    image_flat.sort()

    # 중앙값 리턴
    return image_flat[int(len(image_flat) / 2)]


## 화소영역 처리 함수부
### 블러링
def trans_blurring():
    # 전역 변수
    global outImage

    # 입력 이미지의 높이와 너비 복사하기
    out_height = len(outImage)
    out_width = len(outImage[0])

    # 블러링 마스크 만들기
    blur = 1.0 / 9
    mask = [[blur for _ in range(3)] for _ in range(3)]

    # 마스크 크기 저장
    mask_height = len(mask)
    mask_width = len(mask[0])

    # outImage의 테두리를 127로 두르기 위해 상하좌우 1씩 큰 tempImage 만들기
    tempImage = [[127 for _ in range(out_width + 2)] for _ in range(out_height + 2)]

    # tempImage <- outImage 복사
    for h in range(out_height):
        for w in range(out_width):
            tempImage[h + 1][w + 1] = outImage[h][w]

    # 마스크 적용하기
    for h in range(out_height):
        for w in range(out_width):
            temp = 0.0
            for mh in range(mask_height):
                for mw in range(mask_width):
                    temp += tempImage[h + mh][w + mw] * mask[mh][mw]

            outImage[h][w] = 255 if int(temp) > 255 else 0 if int(temp) < 0 else int(temp)

    # 화면에 출력
    display_image(outImage)


### 엠보싱
def trans_embossing():
    # 전역 변수
    global outImage

    # 입력 이미지의 높이와 너비 복사하기
    out_height = len(outImage)
    out_width = len(outImage[0])

    # 엠보싱 마스크 만들기
    mask = [[-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]]

    # 마스크 크기 저장
    mask_height = len(mask)
    mask_width = len(mask[0])

    # outImage의 테두리를 127로 두르기 위해 상하좌우 1씩 큰 tempImage 만들기
    tempImage = [[127 for _ in range(out_width + 2)] for _ in range(out_height + 2)]

    # tempImage <- outImage 복사
    for h in range(out_height):
        for w in range(out_width):
            tempImage[h + 1][w + 1] = outImage[h][w]

    # 마스크 적용하기
    for h in range(out_height):
        for w in range(out_width):
            temp = 0.0
            for mh in range(mask_height):
                for mw in range(mask_width):
                    temp += tempImage[h + mh][w + mw] * mask[mh][mw]
            # 후처리 (마스크의 총 합이 1이기 때문에)
            temp += 127.0
            outImage[h][w] = 255 if int(temp) > 255 else 0 if int(temp) < 0 else int(temp)

    # 화면에 출력
    display_image(outImage)


### 고주파 샤프닝
def trans_high_freq_sharpening():
    # 전역 변수
    global outImage

    # 입력 이미지의 높이와 너비 복사하기
    out_height = len(outImage)
    out_width = len(outImage[0])

    boundary = -1.0 / 9
    center = 8.0 / 9
    mask = [[boundary for _ in range(3)] for _ in range(3)]
    mask[1][1] = center

    # 마스크 크기 저장
    mask_height = len(mask)
    mask_width = len(mask[0])

    # outImage의 테두리를 127로 두르기 위해 상하좌우 1씩 큰 tempImage 만들기
    tempImage = [[127 for _ in range(out_width + 2)] for _ in range(out_height + 2)]

    # tempImage <- outImage 복사
    for h in range(out_height):
        for w in range(out_width):
            tempImage[h + 1][w + 1] = outImage[h][w]

    # 마스크 적용하기
    for h in range(out_height):
        for w in range(out_width):
            temp = 0.0
            for mh in range(mask_height):
                for mw in range(mask_width):
                    temp += tempImage[h + mh][w + mw] * mask[mh][mw]
            # 원본 영상에 추출한 고주파 값을 더해주어 샤프닝 효과 적용
            temp += outImage[h][w]
            outImage[h][w] = 255 if int(temp) > 255 else 0 if int(temp) < 0 else int(temp)

    # 화면에 출력
    display_image(outImage)


### 가우시안 스무딩
def trans_gausian_smoothing():
    # 전역 변수
    global outImage

    # 입력 이미지의 높이와 너비 복사하기
    out_height = len(outImage)
    out_width = len(outImage[0])

    # 가우시안 스무딩 마스크 만들기
    mask = [[1.0 / 16, 1.0 / 8, 1.0 / 16],
            [1.0 / 8, 1.0 / 4, 1.0 / 8],
            [1.0 / 16, 1.0 / 8, 1.0 / 16]]

    # 마스크 크기 저장
    mask_height = len(mask)
    mask_width = len(mask[0])

    # outImage의 테두리를 127로 두르기 위해 상하좌우 1씩 큰 tempImage 만들기
    tempImage = [[127 for _ in range(out_width + 2)] for _ in range(out_height + 2)]

    # tempImage <- outImage 복사
    for h in range(out_height):
        for w in range(out_width):
            tempImage[h + 1][w + 1] = outImage[h][w]

    # 마스크 적용하기
    for h in range(out_height):
        for w in range(out_width):
            temp = 0.0
            for mh in range(mask_height):
                for mw in range(mask_width):
                    temp += tempImage[h + mh][w + mw] * mask[mh][mw]

            outImage[h][w] = 255 if int(temp) > 255 else 0 if int(temp) < 0 else int(temp)

    # 화면에 출력
    display_image(outImage)


### 1차 미분 에지 검출
def trans_edge_1d_prewitt():
    # 전역 변수
    global outImage

    # 입력 이미지의 높이와 너비 복사하기
    out_height = len(outImage)
    out_width = len(outImage[0])

    # 프리윗 수평, 수직 마스크 만들기
    h_mask = [[-1.0, -1.0, -1.0],
              [0.0, 0.0, 0.0],
              [1.0, 1.0, 1.0]]

    v_mask = [[1.0, 0.0, -1.0],
              [1.0, 0.0, -1.0],
              [1.0, 0.0, -1.0]]

    # 마스크 크기 저장
    mask_height = len(h_mask)
    mask_width = len(h_mask[0])

    # outImage의 테두리를 127로 두르기 위해 상하좌우 1씩 큰 tempImage 만들기
    tempImage = [[127 for _ in range(out_width + 2)] for _ in range(out_height + 2)]

    # tempImage <- outImage 복사
    for h in range(out_height):
        for w in range(out_width):
            tempImage[h + 1][w + 1] = outImage[h][w]

    # 마스크 적용하기
    for h in range(out_height):
        for w in range(out_width):
            temp = 0.0
            for mh in range(mask_height):
                for mw in range(mask_width):
                    # 수직마스크 적용
                    temp += tempImage[h + mh][w + mw] * h_mask[mh][mw]
                    # 수평마스크 적용
                    temp += tempImage[h + mh][w + mw] * v_mask[mh][mw]


            outImage[h][w] = 255 if int(temp) > 255 else 0 if int(temp) < 0 else int(temp)

    # 화면에 출력
    display_image(outImage)


### 2차 미분 에지 검출
def trans_edge_2d_LoG():
    # 전역 변수
    global outImage

    # 입력 이미지의 높이와 너비 복사하기
    out_height = len(outImage)
    out_width = len(outImage[0])

    # 가우시안 마스크 만들기
    mask_gau = [[1.0 / 16, 1.0 / 8, 1.0 / 16],
                [1.0 / 8, 1.0 / 4, 1.0 / 8],
                [1.0 / 16, 1.0 / 8, 1.0 / 16]]

    # 라플라시안 마스크 만들기
    mask_lap = [[-1.0, -1.0, -1.0],
                [-1.0, 8.0, -1.0],
                [-1.0, -1.0, -1.0]]

    # 마스크 크기 저장
    mask_height = len(mask_gau)
    mask_width = len(mask_gau[0])

    # 중앙값 가져오기
    median = get_median(outImage)

    # outImage의 테두리를 127로 두르기 위해 상하좌우 1씩 큰 tempImage 만들기
    tempImage = [[127 for _ in range(out_width + 2)] for _ in range(out_height + 2)]

    # tempImage <- outImage 복사
    for h in range(out_height):
        for w in range(out_width):
            tempImage[h + 1][w + 1] = outImage[h][w]

    # 가우시안 스무딩 마스크 적용
    for h in range(out_height):
        for w in range(out_width):
            temp = 0.0
            for mh in range(mask_height):
                for mw in range(mask_width):
                    # 가우시안 스무딩 적용
                    temp += tempImage[h + mh][w + mw] * mask_gau[mh][mw]
            outImage[h][w] = 255 if int(temp) > 255 else 0 if int(temp) < 0 else int(temp)

    # 라플라시안 마스크 적용
    for h in range(out_height):
        for w in range(out_width):
            temp = 0.0
            for mh in range(mask_height):
                for mw in range(mask_width):
                    temp += tempImage[h + mh][w + mw] * mask_lap[mh][mw]
            # 후처리 (중앙값)
            temp += median
            outImage[h][w] = 255 if int(temp) > 255 else 0 if int(temp) < 0 else int(temp)

    # 화면에 출력
    display_image(outImage)


## 기하학 처리 함수부

### 회전
def trans_rotate_no_cutting():
    # 전역 변수
    global inImage, outImage

    # 각도 값 입력
    degree = askinteger("회전 각도 입력", "-90~90 입력", maxvalue=90, minvalue=-90)

    # 입력한 각도를 라디안으로 변환
    radian = degree * 3.141592 / 180.0

    # 배경의 높이와 너비 초기화
    background_height = 0
    background_width = 0

    in_height = len(inImage)
    in_width = len(inImage[0])

    # 배경의 크기 계산
    background_height = int(math.cos(radian) * in_height + math.cos(radian) * in_width)
    background_width = int(math.cos(radian) * in_height + math.cos(radian) * in_width)

    # 홀수 크기 처리
    if (background_height % 2):
        background_height += 1

    if (background_width % 2):
        background_width += 1

    # 배경의 높이와 너비를 복사본의 높이와 너비로 복사
    out_height = background_height
    out_width = background_width

    # outImage 검은색으로 초기화
    outImage = [[0 for _ in range(out_width)] for _ in range(out_height)]

    # 입력 영상의 중심점 잡기 (회전 후 좌표 보정용)
    in_center_height = int(in_height / 2)
    in_center_width = int(in_width / 2)

    # 출력 영상의 중심점 잡기 (회전 후 좌표 보정용)
    out_center_height = int(out_height / 2)
    out_center_width = int(out_width / 2)

    # 영상 회전(백워딩) 및 중심 좌표 보정
    for h in range(out_height):
        for w in range(out_width):
            xd = h
            yd = w
            # x = xcosΘ - ysinΘ  , y = xsinΘ + ycosΘ
            # 출력 이미지 중앙점 보정
            xs = math.cos(radian) * (xd - out_center_height) - math.sin(radian) * (yd - out_center_width)
            ys = math.sin(radian) * (xd - out_center_height) + math.cos(radian) * (yd - out_center_width)
            # 원본 이미지 중앙점 보정
            xs += in_center_height
            ys += in_center_width
            if ((0 <= xs and xs < in_height) and (0 <= ys and ys < in_width)):
                outImage[int(xd)][int(yd)] = inImage[int(xs)][int(ys)]

    display_image(outImage)


### 좌우 이동
def trans_move_rl():
    # 전역 변수
    global inImage, in_height, in_width

    # 입력 이미지의 높이와 너비 복사하기
    temp_height = in_height
    temp_width = in_width

    # 이동 값 입력
    iVal = askinteger("이동할 크기 입력(좌우)", "-"+str(temp_height)+"~"+str(temp_height)+" 입력", maxvalue=temp_height, minvalue=-temp_height)

    temp_width += abs(iVal)

    tempImage = [[0 for _ in range(temp_width)]for _ in range(temp_height)]

    # 이동 안 잘림
    # 입력한 정수가 양수일 때
    if (iVal > 0):
        for h in range(in_height):
            for w in range(in_width):
                pixel_moved = w + iVal - 1
                tempImage[h][pixel_moved] = inImage[h][w]
    # 입력한 정수가 0 이하일 때
    elif(iVal <=0 ):
        for h in range(in_height):
            for w in range(in_width):
                tempImage[h][w] = inImage[h][w]

    # 화면 출력
    display_image(tempImage)


### 상하 이동
def trans_move_ud():
    # 전역 변수
    global inImage, in_height, in_width

    # 입력 이미지의 높이와 너비 복사하기
    temp_height = in_height
    temp_width = in_width

    # 이동 값 입력
    iVal = askinteger("이동할 크기 입력(상하)", "-" + str(temp_height) + "~" + str(temp_height) + " 입력", maxvalue=temp_height,
                      minvalue=-temp_height)

    temp_height += abs(iVal)

    tempImage = [[0 for _ in range(temp_width)] for _ in range(temp_height)]

    # 이동 안 잘림
    # 입력한 정수가 양수일 때
    if (iVal > 0):
        for h in range(in_height):
            for w in range(in_width):
                pixel_moved = h + iVal - 1
                tempImage[pixel_moved][w] = inImage[h][w]
    # 입력한 정수가 0 이하일 때
    elif (iVal <= 0):
        for h in range(in_height):
            for w in range(in_width):
                tempImage[h][w] = inImage[h][w]

    # 화면 출력
    display_image(tempImage)


### 상하 미러링
def trans_mirror_h():
    # 전역 변수
    global outImage

    # 입력 이미지의 높이와 너비 복사하기
    out_height = len(outImage)
    out_width = len(outImage[0])

    # 상하 미러링
    for h in range(out_height):
        for w in range(out_width):
            outImage[out_height - 1 - h][w] = inImage[h][w]
            outImage[h][w] = inImage[out_height - 1 - h][w]

    # 화면 출력
    display_image(outImage)


### 좌우 미러링
def trans_mirror_v():
    # 전역 변수
    global outImage

    # 입력 이미지의 높이와 너비 복사하기
    out_height = len(outImage)
    out_width = len(outImage[0])

    # 좌우 미러링
    for h in range(out_height):
        for w in range(out_width):
            outImage[h][out_width - 1 - w] = inImage[h][w]
            outImage[h][w] = inImage[h][out_width - 1 - w]

    # 화면 출력
    display_image(outImage)


### 확대
def trans_zoom_in():
    # 전역 변수
    global inImage, outImage
    global in_height, in_width
    global out_height, out_width

    # 몇 배 늘릴까
    scale = 2
    out_height = in_height * scale
    out_width = in_width * scale
    outImage = [[0 for _ in range(out_width)] for _ in range(out_height)]
    # 확대 백워딩
    for h in range(out_height):
        for w in range(out_width):
            inH = int(h / scale)
            inW = int(w / scale)
            if((0<= inH <in_height) and (0<= inW <in_width)):
                outImage[h][w] = inImage[inH][inW]
    # 화면 출력
    display_image(outImage)


### 축소
def trans_zoom_out():
    # 전역 변수
    global inImage, outImage
    global in_height, in_width
    global in_width, out_width

    # 몇 배 줄일까
    scale = 2
    out_height = int(in_height / scale)+1
    out_width = int(in_width / scale)+1
    outImage = [[0 for _ in range(out_width)] for _ in range(out_height)]
    # 축소 포워딩
    for h in range(in_height):
        for w in range(in_width):
            outH = int(h / scale)
            outW = int(w / scale)
            if ((0 <= outH < out_height) and (0 <= outW < out_width)):
                outImage[outH][outW] = inImage[h][w]

    # 화면 출력
    display_image(outImage)



#------------------------------------------------------#
## 윈도우 기본
window = Tk()       # root 라는 이름으로도 사용
window.geometry("512x512")
window.title("Gray Scale 영상처리")
#window.resizable(width=False,height=False)  # 크기 조절 안됨

#------------------------------------------------------#

## 캔버스 (점 찍기 안됨)
canvas = Canvas(window,height=256,width=256,bg="yellow")

#------------------------------------------------------#

## 페이퍼 (점 찍기 됨)
### 중앙점 설정 중요
paper = PhotoImage(height=256,width=256)
canvas.create_image((256/2,256/2),image=paper,state="normal")

#------------------------------------------------------#

### 메인 메뉴
main_menu = Menu(window)
window.config(menu=main_menu)

#------------------------------------------------------#

#### 상위 메뉴 = 펼쳐 진다 (파일메뉴 - add_cascade)

##### 파일 메뉴
file_menu = Menu(main_menu)
main_menu.add_cascade(label="파일",menu=file_menu)

##### 화소점처리(영상처리) 메뉴
pixel_dot_menu = Menu(main_menu)
main_menu.add_cascade(label="화소점",menu=pixel_dot_menu)

##### 히스토그램 메뉴
hist_menu = Menu(main_menu)
main_menu.add_cascade(label="히스토그램",menu=hist_menu)

##### 화소영역처리(영상처리) 메뉴
pixel_mask_menu = Menu(main_menu)
main_menu.add_cascade(label="화소영역",menu=pixel_mask_menu)

##### 기하학처리(영상처리) 메뉴
geometry_menu = Menu(main_menu)
main_menu.add_cascade(label="기하학",menu=geometry_menu)

##### 영상 통계 메뉴
statistics_menu = Menu(main_menu)
main_menu.add_cascade(label="영상통계",menu=statistics_menu)
#------------------------------------------------------#

#### 하위 메뉴 = 실행 된다 (파일메뉴 - add_command)

##### 파일 하위 메뉴
file_menu.add_command(label="열기",command=open_image)
file_menu.add_command(label="저장",command=save_image)
file_menu.add_separator()
file_menu.add_command(label="종료",command=exit_app)

##### 화소점처리(영상처리) 하위 메뉴
pixel_dot_menu.add_command(label="초기화",command=equal_image)
pixel_dot_menu.add_command(label="명도조절(+-)",command=trans_brightness_pm)
pixel_dot_menu.add_command(label="명도조절(*)",command=trans_brightness_mul)
pixel_dot_menu.add_command(label="명도조절(/)",command=trans_brightness_div)
pixel_dot_menu.add_command(label="감마보정",command=trans_gamma)
pixel_dot_menu.add_command(label="흑백처리",command=trans_bow)
pixel_dot_menu.add_command(label="반전처리",command=trans_reverse)
pixel_dot_menu.add_separator()
pixel_dot_menu.add_command(label="AND",command=trans_and)
pixel_dot_menu.add_command(label="OR",command=trans_or)
pixel_dot_menu.add_command(label="XOR",command=trans_xor)
pixel_dot_menu.add_separator()
pixel_dot_menu.add_command(label="파라CAP",command=trans_para_cap)
pixel_dot_menu.add_command(label="파라CUP",command=trans_para_cup)
pixel_dot_menu.add_separator()

##### 히스토그램 하위 메뉴
hist_menu.add_command(label="히스토그램 스트레치",command=trans_hist_stretch)
hist_menu.add_command(label="히스토그램 엔드인",command=trans_hist_endin_stretch)
hist_menu.add_command(label="히스토그램 평활화",command=trans_hist_equal)

##### 화소영역처리(영상처리) 하위 메뉴
pixel_mask_menu.add_command(label="블러링",command=trans_blurring)
pixel_mask_menu.add_command(label="엠보싱",command=trans_embossing)
pixel_mask_menu.add_command(label="고주파 샤프닝",command=trans_high_freq_sharpening)
pixel_mask_menu.add_command(label="가우시안 스무딩",command=trans_gausian_smoothing)
pixel_mask_menu.add_command(label="1차원 에지 프리윗",command=trans_edge_1d_prewitt)
pixel_mask_menu.add_command(label="2차원 에지 LoG",command=trans_edge_2d_LoG)

##### 기하학처리 하위 메뉴
geometry_menu.add_command(label="회전",command=trans_rotate_no_cutting)
geometry_menu.add_command(label="좌우이동",command=trans_move_rl)
geometry_menu.add_command(label="상하이동",command=trans_move_ud)
geometry_menu.add_command(label="좌우대칭",command=trans_mirror_v)
geometry_menu.add_command(label="상하대칭",command=trans_mirror_h)
geometry_menu.add_command(label="확대",command=trans_zoom_in)
geometry_menu.add_command(label="축소",command=trans_zoom_out)

##### 영상통계 하위 메뉴
statistics_menu.add_command(label="평균",command=get_average)
statistics_menu.add_command(label="중앙값",command=get_median)
#------------------------------------------------------#

canvas.pack()
window.mainloop()