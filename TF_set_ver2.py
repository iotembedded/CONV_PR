import os
import numpy as np
from PIL import Image, ImageOps
import math

#import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import backend as K
# from tensorflow.keras.utils import Sequence
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix

from numpy import expand_dims

#숫자를 문자열로 list 저장
num = list(map(str, range(1, 20)))
human_num = list(map(str, range(1, 460)))

#경로
Path = "E:/4-여름/연구과제(반도체Leak검출)/지문 이미지 데이터 과제/labelingData/"

#숫자 인덱스에 따라 경로(문자열) 반환
def Path_f(hn_f, n_f):
    path_all = Path + human_num[hn_f-1] + "_" + num[n_f-1]+".bmp"
    return path_all

# -------------------------------- True Set ----------------------------------------- #
def TrueSet():
    s_data = 0
    num_v = [[0 for col in range(1, 461)] for row in range(1, 461)]
    data = []

    # 컴비네이션을 위한 개수세기 알고리즘
    for i in range(1, 460):
        for j in range(1, 20):
            if os.path.isfile(Path_f(i, j)) == True: #해당 파일이 있으면?
                data.append(j)                       #인덱스 값 data 변수에 저장
        num_v[i] = list(combinations(data, 2)) #한 명 끝나면 컴비네이션 값 num_v에 저장
        data.clear()                           #data값 초기화
    # 텐서 초기화
    # 459명 x 15C2(105) = 48,195
    PERSON = 10
    data_X = np.zeros((PERSON * 105, 336, 258, 2))
    data_Y = np.zeros((PERSON * 105))
    # num_v[] 사람 명수 num_v[][] 사람 별 컴비네이션 개수
    # 텐서 대입
    print("---------TRUE SET : START---------")
    cnt=0
    for i in range(1, PERSON):
        for j in range(1, 20):
            if os.path.isfile(Path_f(i, num_v[i][j][0])) and os.path.isfile(Path_f(i, num_v[i][j][1]))== True: # 경로에 파일이 있을 시에만 동작
                # 지정된 컴비네이션에 따른 경로 설정
                x1 = Path_f(i, num_v[i][j][0])
                x2 = Path_f(i, num_v[i][j][1])
                im1 = Image.open(x1)
                im2 = Image.open(x2)
                im1 = np.array(im1)
                im2 = np.array(im2)

                data_X[cnt, :, :, 0] = im1
                data_X[cnt, :, :, 1] = im2
                data_Y[cnt] = 1
                cnt += 1
                print("INDEX :", cnt)
    print("---------TRUE SET : END---------")

# -------------------------------- False Set ----------------------------------------- #
def FalseSet():
    s_data = 0
    cnt_list = [0 for i in range(1, 461)]
    num_v1 = []
    data = []

    # 컴비네이션을 위한 개수세기 알고리즘
    for i in range(1, 460):
        if os.path.isfile(Path_f(i, 1)) == True:
            data.append(i)
    num_v1 = list(combinations(data, 2))
    data.clear()

    # 텐서 초기화
    # personC2 x 15 x 15 = 48,195
    PERSON = 10
    data_X = np.zeros((PERSON*15*15, 336, 258, 2))
    data_Y = np.zeros((PERSON*15*15))
    # 텐서 대입
    print("---------FALSE SET : START---------")
    cnt =0
    for i in range(1, PERSON):
        for j in range(1, 20):
            for k in range(1, 20):
                # 경로에 파일 있을 시
                if os.path.isfile(Path_f(num_v1[i][0], j)) and os.path.isfile(Path_f(num_v1[i][1], k)) == True:
                    x1 = Path_f(num_v1[i][0], j)
                    x2 = Path_f(num_v1[i][1], k)
                    im1 = Image.open(x1)
                    im2 = Image.open(x2)
                    im1 = np.array(im1)
                    im2 = np.array(im2)

                    data_X[cnt, :, :, 0] = im1
                    data_X[cnt, :, :, 1] = im2
                    data_Y[cnt] = 1
                    cnt += 1
                    print("INDEX :", cnt)
    print("---------FALSE SET : END---------")
    
#TrueSet()
#FalseSet()
