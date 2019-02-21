# 利用dlib裁剪人脸用于生成数据集

import sys
import os
import cv2
import dlib
import shutil

def get_face_only():
    # 截取脸部图片放入相同文件夹
    input_dir = '.\\data\\lfw'
    output_dir = '.\\data\\faces'
    # 新建输出文件夹
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # dlib的人脸提取器
    detector = dlib.get_frontal_face_detector()
    size = 96
    index = 1
    for (path, dirnames, filenames) in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('jpg'):
                print('Being process picture %s' % index)
                img_path = path + '/' + filename
                img = cv2.imread(img_path)
                # 转化为灰度图
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # 人脸检测
                dets = detector(gray_img, 1)

                for i, d in enumerate(dets):
                    x1 = d.top() if d.top() > 0 else 0
                    x2 = d.bottom() if d.bottom() > 0 else 0
                    y1 = d.left() if d.left() > 0 else 0
                    y2 = d.right() if d.right() > 0 else 0
                    face = img[x1:x2, y1:y2]
                    # 重新调整图片尺寸,基于局部像素的重采样,默认是双线性插值
                    face = cv2.resize(face, (size, size), interpolation=cv2.INTER_AREA)
                    cv2.imshow('image', face)
                    # 保存图片
                    save_path = output_dir + "\\" + path.split('\\')[3]
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                    cv2.imwrite(save_path + "\\" + str(index) + ".jpg", face)
                    # print(path.split('\\')[3])
                    index += 1
                key = cv2.waitKey(30) & 0xff
                if key == 27:
                    sys.exit()

# 将生成的人脸数据复制到face文件夹中
def copy_face():
    input_dir = '.\\data\\faces'
    output_dir = '.\\data\\only_faces'
    # 新建输出文件夹
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for (path, dirnames, filenames) in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('jpg'):
                img_path = path + '/' + filename
                print(img_path)
                shutil.copy(img_path, output_dir)

def get_CASIA_face():
    # 截取脸部图片放入相同文件夹
    input_dir = '.\data\CASIA-FaceV5\CASIA-FaceV5 (400-499)'
    output_dir = '.\\data\\only_faces'
    # 新建输出文件夹
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # dlib的人脸提取器
    detector = dlib.get_frontal_face_detector()
    size = 96
    index = 15954
    for (path, dirnames, filenames) in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('bmp'):
                print('Being process picture %s' % index)
                img_path = path + '/' + filename
                img = cv2.imread(img_path)
                # 转化为灰度图
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # 人脸检测
                dets = detector(gray_img, 1)

                for i, d in enumerate(dets):
                    x1 = d.top() if d.top() > 0 else 0
                    x2 = d.bottom() if d.bottom() > 0 else 0
                    y1 = d.left() if d.left() > 0 else 0
                    y2 = d.right() if d.right() > 0 else 0
                    face = img[x1:x2, y1:y2]
                    # 重新调整图片尺寸,基于局部像素的重采样,默认是双线性插值
                    face = cv2.resize(face, (size, size), interpolation=cv2.INTER_AREA)
                    cv2.imshow('image', face)
                    # 保存图片
                    cv2.imwrite(output_dir + "\\" + str(index) + ".jpg", face)
                    # print(path.split('\\')[3])
                    index += 1
                key = cv2.waitKey(30) & 0xff
                if key == 27:
                    sys.exit()
# 获取celeba的人脸数据
def get_CELEBA_face():
    # 截取脸部图片放入相同文件夹
    input_dir = '.\data\celeba'
    output_dir = '.\\data\\only_faces'
    # 新建输出文件夹
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # dlib的人脸提取器
    detector = dlib.get_frontal_face_detector()
    size = 96
    index = 16444
    for (path, dirnames, filenames) in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('jpg'):
                print('Being process picture %s' % index)
                img_path = path + '/' + filename
                img = cv2.imread(img_path)
                # 转化为灰度图
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # 人脸检测
                dets = detector(gray_img, 1)
                for i, d in enumerate(dets):
                    x1 = d.top() if d.top() > 0 else 0
                    x2 = d.bottom() if d.bottom() > 0 else 0
                    y1 = d.left() if d.left() > 0 else 0
                    y2 = d.right() if d.right() > 0 else 0
                    face = img[x1:x2, y1:y2]
                    # 重新调整图片尺寸,基于局部像素的重采样,默认是双线性插值
                    face = cv2.resize(face, (size, size), interpolation=cv2.INTER_AREA)
                    cv2.imshow('image', face)
                    # 保存图片
                    cv2.imwrite(output_dir + "\\" + str(index) + ".jpg", face)
                    # print(path.split('\\')[3])
                    index += 1
                    if index >= 50000:
                        os.system('pause')
                key = cv2.waitKey(30) & 0xff
                if key == 27:
                    sys.exit()
if __name__ == '__main__':
    # get_face_only()
    # copy_face()
    # get_CASIA_face()
    get_CELEBA_face()













