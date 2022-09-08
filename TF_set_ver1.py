import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import os
import itertools
import random
import numpy as np
import time

#경로
Path = "E:/4-여름/연구과제(반도체Leak검출)/지문 이미지 데이터 과제/labelingData/"

#숫자를 문자열로 list 저장
num = list(map(str, range(1, 19)))
human_num = list(map(str, range(1, 460)))
img = []
img_t = []
Path_num = []
num_v = []
tf = transforms.ToTensor()


def change_num():
    number_random = random.sample(num, 2)  # a라는 리스트에서 2개를 랜덤으로 추출
    return number_random

#입력 숫자에 따른 문자열(경로) 반환
def Path_f(hn_f, n_f):
    path_all = Path + human_num[hn_f-1] + "_" + num[n_f-1]+".bmp"
    return path_all

#n번 인물에 따른 지문 Trueset을 만들어 주는 함수
def True_set_make(hn_f1):
    num_v = random.sample(range(1, 19), 2)
    x1 = Path_f(hn_f1, num_v[0])
    x2 = Path_f(hn_f1, num_v[1])

    # 경로에 해당하는 파일이 있을 시 다음으로 진행
    s_data=0
    while True:
        for j in range(1, 19):
            if os.path.isfile(Path_f(hn_f1, j)) == False:
                s_data += 1
        if s_data == 18:
            hn_f1 += 1
            s_data =0
        else:
            break

    #만약 해당 경로에 파일이 없을 시(1_x.png  해당 x가 없어 파일이 검색 안될 때)
    while os.path.isfile(x1) == False or os.path.isfile(x2) == False:
        num_v = random.sample(range(1, 19), 2)
        x1 = Path_f(hn_f1, num_v[0])
        x2 = Path_f(hn_f1, num_v[1])

    #이미지 결합
    img1 = Image.open(x1)
    img2 = Image.open(x2)
    img_t1 = tf(img1)
    img_t2 = tf(img2)
    result = torch.cat([img_t1, img_t2], dim=2)
    return result

#n번 인물에 따른 지문 Falseset을 만들어 주는 함수
def False_set_make(hn_f2):
    # Trueset 해당 인물과 다른 인물이면 통과
    while True:
        num_f = random.sample(range(1, 460), 2)

        if num_f[0] != hn_f2 and num_f[1] != hn_f2:
            break

    num_v = random.sample(range(1, 19), 2)

    x1 = Path_f(num_f[0], num_v[0])
    x2 = Path_f(num_f[1], num_v[1])

    # 만약 해당 경로에 파일이 없을 시(뒤의 숫자로 인해 파일이 검색 안될 때)
    while os.path.isfile(x1) == False or os.path.isfile(x2) == False:
        num_v = random.sample(range(1, 19), 2)
        while True:
            num_f = random.sample(range(1, 460), 2)

            if num_f[0] != hn_f2 and num_f[1] != hn_f2:
                break
        x1 = Path_f(num_f[0], num_v[0])
        x2 = Path_f(num_f[1], num_v[1])

    img1 = Image.open(x1)
    img2 = Image.open(x2)
    img_t1 = tf(img1)
    img_t2 = tf(img2)
    result = torch.cat([img_t1, img_t2], dim=2)
    return result

# 만들고자하는 숫자 시작과 끝
def True_set_result(start, end):
    True_set = []

    for i in range(start, end):
        if i == 1:
            True_set = True_set_make(1)
            print(i)
        True_set = torch.cat([True_set, True_set_make(i + 1)], dim=0)
        print(i + 1)
    print("True set 완성")
    print(True_set.shape)

# 만들고자하는 숫자 시작과 끝
def False_set_result(start, end):
    False_set = []

    for i in range(start, end):
        if i == 1:
            False_set = False_set_make(1)
            print(i)
        False_set = torch.cat([False_set, False_set_make(i + 1)], dim=0)
        print(i + 1)
    print("False set 완성")
    print(False_set.shape)

#1~459까지의 Tensor Trueset 만드는 것
#True_set_result(1, 460)
#1~459까지의 Tensor Falseset 만드는 것
#False_set_result(1, 460)
