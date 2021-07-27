# [1] 필요한 라이브러리 불러오기
import cv2, dlib, sys #cv2 : 이미지 처리 라이브러리 / dlib : 얼굴 인식을 위한 이미 처리 라이브러리 / numpy : 행렬 연산 라이브러리
import numpy as np

# [2] 사용할 비디오 불러오기
scaler = 0.3  # 크기를 3/10 으로 맞춘다.

detector = dlib.get_frontal_face_detector() # dlib.get_frontal_face_detector() : 일굴 디텍터 모듈을 초기화한다.
predictor = dlib.shape_predictor() # dlib.shape_predictor() : 얼굴의 특징점 모듈을 초기화한다. / 머신러닝으로 학습되어있는 모델이기 때문에 모델 파일을 다운받는다.


cap = cv2.VideoCapture('girl.mp4') # VideoCapture:  .mp4 동영상 파일 로드 / 만약 파일 이름 대신 0을 넣으면 웹캠으로 얼굴 인식이 가능하다.

# [3] 비디오 불러오기
while True: # 반복문을 써서 비디오가 끝날 때까지 frame 단위로 계속 불러온다. / 만약 frame이 없으면 break로 종료한다. 
    ret, img = cap.read() # cap.read() : 동영상 파일에서 frame(ret) 단위로 읽는다.
    if not ret:
        break

    img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))  # cv2.resize(img, dsize) : img를 dsize 크기로 조절함
    ori = img.copy()  # 원본 이미지를 ori라는 변수에 저장해둔다.


#[4] 얼굴을 인식하기
    faces = detector(img) # detector(img) : img에서 모든 얼굴을 찾아서 자동으로 인식한다.
    face = faces[0] # faces에는 여러 얼굴이 인식되기 때문에 그 중에 하나(0번째)의 얼굴만 가져온다. 

#[5] 얼굴 특징점(눈, 코, 입 등등) 추출하기
    dlib_shape = predictor(img, face)  # predictor(img, face) : img의 face 영역안의 얼굴의 특징점을 추출한다. / 이미지와 앞에서 구했던 얼굴 영역을 가지고 추출한다.
    shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])  # dlib 객체를 numpy 객체로 변환 / dlib shape를 리턴받아서 연산을 쉽게 하기 위해 numpy array로 바꾸어서 shape_2d 변수에 저장한다.

#[6] 얼굴의 크기와 중심 구하기
    top_left = np.min(shape_2d, axis=0) # np.min() : 최솟값 찾기 / 좌상단 점을 구한다. 
    bottom_right = np.max(shape_2d, axis=0) # np.max() : 최대값 찾기 /  우하단 점을 구한다.

#[7] 얼굴 영역 박스로 표시하기
    img = cv2.rectangle(img, pt1=(face.left(), face.top()), pt2=(face.right(), face.bottom()), color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA) # cv2.rectangle() : 직사각형을 그린다. / pt1 : 좌상단, pt2 : 우하단 / color : 흰색 / thickness(두께) : 2 / lineType : LINE_AA(계단 현상(사선으로 그렸을 때 픽셀 깨짐 현상)이 없는 매끄러운 선)

#[8] 얼굴 특징점 표시하기
    for s in shape_2d: # 얼굴 특징점의 개수는 총 68개이고 그것을 circle() 함수를 이용하여 얼굴에 그린다. 
        cv2.circle(img, center=tuple(s), radius=1, colo=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA) # cv2.circle() : 원 그리기

#[9] 좌상단과 우하단의 점을 이미지에 표시하기
    cv2.circle(img, center=tuple(top_left), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA) # 특징점과 동일하게 circle()을 사용하고 색상과 좌표만 다르다. 
    cv2.circle(img, center=tuple(bottom_right), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA) # 특징점과 동일하게 circle()을 사용하고 색상과 좌표만 다르다. 
    

#[10] 결과 출력하기
    cv2.imshow('img', img) # img에 저장해둔 프레임 단위의 이미지를 읽어서 img 윈도우 창에 띄운다.
    cv2.waitKey(1) # 1 m/s만큼 대기한다. 이것을 넣어야 영상이 제대로 보인다. 
