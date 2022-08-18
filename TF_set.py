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

Path = "E:/4-여름/연구과제(반도체Leak검출)/지문 이미지 데이터 과제/labelingData/"

num = list(map(str, range(1, 19)))
human_num = list(map(str, range(1, 460)))
img = []
img_t = []
Path_num = []
num_v = []
tf = transforms.ToTensor()
# def tensorTOimage():
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print('Using {} device'.format(device))
#
#     input_image = torch.rand(3,128,128)
#     print(input_image.size())
#
#     tf = transforms.ToPILImage()
#     img = tf(input_image)
#     img.show()
#
# def imageTOtensor(num):
#     Path_num = Path + human_num + "_"+num+".bmp"
#     img = Image.open(Path_num)
#     tf = transforms.ToTensor()
#     img_t = tf(img)
#     print(img_t)

def change_num():
    number_random = random.sample(num, 2)  # a라는 리스트에서 2개를 랜덤으로 추출
    return number_random

def Path_f(hn_f, n_f):
    path_all = Path + human_num[hn_f-1] + "_" + num[n_f-1]+".bmp"
    return path_all

def True_set_make(hn_f1):
    num_v = random.sample(range(1, 19), 2)
    x1 = Path_f(hn_f1, num_v[0])
    x2 = Path_f(hn_f1, num_v[1])

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

    #만약 해당 경로에 파일이 없을 시(뒤의 숫자로 인해 파일이 검색 안될 때)
    while os.path.isfile(x1) == False or os.path.isfile(x2) == False:
        num_v = random.sample(range(1, 19), 2)
        x1 = Path_f(hn_f1, num_v[0])
        x2 = Path_f(hn_f1, num_v[1])

    img1 = Image.open(x1)
    img2 = Image.open(x2)
    img_t1 = tf(img1)
    img_t2 = tf(img2)
    result = torch.cat([img_t1, img_t2], dim=2)
    return result

def False_set_make(hn_f2):
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

# imageTOtensor('1', '1')
#입력받은 숫자를 저장할 리스트를 만들어준다
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

False_set_result(1, 460)
#print(a.shape)
# print(human_num[458])
# # print(range(0, 459)[458])
# a = torch.cat([True_set_make(0), True_set_make(1)], dim=0)
# a = torch.cat([a, True_set_make(2)], dim=0)
# a = torch.cat([a, True_set_make(3)], dim=0)
# a = torch.cat([a, True_set_make(4)], dim=0)
# a = torch.cat([a, True_set_make(5)], dim=0)
# a = torch.cat([a, True_set_make(6)], dim=0)
# a = torch.cat([a, True_set_make(7)], dim=0)
# a = torch.cat([a, True_set_make(8)], dim=0)
# a = torch.cat([a, True_set_make(9)], dim=0)
# a = torch.cat([a, True_set_make(10)], dim=0)
# a = torch.cat([a, True_set_make(11)], dim=0)
#print(a.shape)


#
# print(True_set.shape)

# Path_num = Path + human_num[1] + "_" + num[1]+".bmp"
# img = Image.open(Path_num)
# img_t[1] = tf(img)
#print(tf[1])
