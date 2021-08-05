# -*- coding: utf-8 -*-
import shutil
import dlib  # 人脸识别的库dlib
import numpy as np  # 数据处理的库numpy
import wx  # 构造显示界面的GUI
import wx.xrc
import wx.adv
# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import datetime, time
import math
import os
import pandas as pd
import winsound  # 系统音效
from playsound import playsound  # 音频播放
import csv  # 存入表格
import time
import sys
import numpy as np  # 数据处理的库 numpy
from cv2 import cv2 as cv2  # 图像处理的库 OpenCv
import pandas as pd  # 数据处理的库 Pandas
from skimage import io
import socket
import codecs
## Class Fatigue_detecting
###########################################################################

COVER = 'G:/pycharm project/python project/face detecting/images/camera.png'


facerec = dlib.face_recognition_model_v1(
    "G:/pycharm project/python project/face detecting/model/dlib_face_recognition_resnet_model_v1.dat")
# 用来存放所有录入人脸特征的数组
# the array to save the features of faces in the database
features_known_arr = []

detector = dlib.get_frontal_face_detector()

face_rec = dlib.face_recognition_model_v1("G:/pycharm project/python project/face detecting/model/dlib_face_recognition_resnet_model_v1.dat")

predictor = dlib.shape_predictor("G:/pycharm project/python project/face detecting/model/shape_predictor_68_face_landmarks.dat")






# """
# client
#     connect()
#     recv()
#     send()
#     sendall()
# """
# # 创建套接字，绑定套接字到本地IP与端口
# sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# # address = ('10.1.156.82', 8001)
# sk.connect(('10.1.156.82', 8001))
# inp = '030300000002c5e9'
# while True:
#     if inp == 'exit':
#         print("exit")
#         break
#     # 默认编码为十六进制编码
#     sk.send(codecs.decode(inp, 'hex'))
#     # 每2秒读取以此数据
#     time.sleep(2)
#     # 每次接受1024字节的数据
#     result = sk.recv(1024)
#     result = codecs.encode(result, 'hex')
#     r = bytes(result).decode('utf-8')
#     shidu = int(r[6:10], 16) / 100
#     wendu = int(r[10:14], 16) / 100
#     print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
#     print("温度：%s，湿度：%s\n" % (wendu, shidu))
sk.close()
def return_128d_features(path_img):
    im_rd = io.imread(path_img)
    img_gray = cv2.cvtColor(im_rd, cv2.COLOR_BGR2RGB)
    faces = detector(img_gray, 1)

    print("%-40s %-20s" % ("检测到人脸的图像 / image with faces detected:", path_img), '\n')

    # 因为有可能截下来的人脸再去检测，检测不出来人脸了
    # 所以要确保是 检测到人脸的人脸图像 拿去算特征
    if len(faces) != 0:
        shape = predictor(img_gray, faces[0])
        face_descriptor = face_rec.compute_face_descriptor(img_gray, shape)
    else:
        face_descriptor = 0
        print("no face")

    return face_descriptor

def return_features_mean_personX(path_faces_personX):
    features_list_personX = []
    photos_list = os.listdir(path_faces_personX)
    if photos_list:
        for i in range(len(photos_list)):
            # 调用return_128d_features()得到128d特征
            print("%-40s %-20s" % ("正在读的人脸图像 / image to read:", path_faces_personX + "/" + photos_list[i]))
            features_128d = return_128d_features(path_faces_personX + "/" + photos_list[i])
            #  print(features_128d)
            # 遇到没有检测出人脸的图片跳过
            if features_128d == 0:
                i += 1
            else:
                features_list_personX.append(features_128d)
    else:
        print("文件夹内图像文件为空 / Warning: No images in " + path_faces_personX + '/', '\n')

    # 计算 128D 特征的均值
    # N x 128D -> 1 x 128D
    if features_list_personX:
        features_mean_personX = np.array(features_list_personX).mean(axis=0)
    else:
        features_mean_personX = '0'

    return features_mean_personX
path_images_from_camera = "G:/pycharm project/python project/face detecting/pictures/people/"
class Fatigue_detecting(wx.Frame):

    def __init__(self, parent, title):
        wx.Frame.__init__(self, parent, id=wx.ID_ANY, title=title, pos=wx.DefaultPosition, size=wx.Size(873, 535),
                          style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)
        # wx.Frame.__init__(self, parent, id=wx.ID_ANY, title=title, pos=wx.DefaultPosition, size=wx.Size(900, 700),style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)
        self.SetSizeHints(wx.DefaultSize, wx.DefaultSize)
        self.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_MENU))

        bSizer1 = wx.BoxSizer(wx.VERTICAL)
        bSizer2 = wx.BoxSizer(wx.HORIZONTAL)
        bSizer3 = wx.BoxSizer(wx.VERTICAL)

        self.m_animCtrl1 = wx.adv.AnimationCtrl(self, wx.ID_ANY, wx.adv.NullAnimation, wx.DefaultPosition,
                                                wx.DefaultSize, wx.adv.AC_DEFAULT_STYLE)
        bSizer3.Add(self.m_animCtrl1, 1, wx.ALL | wx.EXPAND, 5)
        bSizer2.Add(bSizer3, 9, wx.EXPAND, 5)
        bSizer4 = wx.BoxSizer(wx.VERTICAL)
        sbSizer1 = wx.StaticBoxSizer(wx.StaticBox(self, wx.ID_ANY, u"parameters setting"), wx.VERTICAL)
        sbSizer2 = wx.StaticBoxSizer(wx.StaticBox(sbSizer1.GetStaticBox(), wx.ID_ANY, u"video source"), wx.VERTICAL)
        gSizer1 = wx.GridSizer(0, 2, 0, 8)
        m_choice1Choices = [u"camera_0", u"camera_1", u"camera_2"]
        self.m_choice1 = wx.Choice(sbSizer2.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.Size(90, 25),
                                   m_choice1Choices, 0)
        self.m_choice1.SetSelection(0)
        gSizer1.Add(self.m_choice1, 0, wx.ALL, 5)
        self.camera_button1 = wx.Button(sbSizer2.GetStaticBox(), wx.ID_ANY, u"start detecting", wx.DefaultPosition,
                                        wx.Size(90, 25), 0)
        gSizer1.Add(self.camera_button1, 0, wx.ALL, 5)
        self.vedio_button2 = wx.Button(sbSizer2.GetStaticBox(), wx.ID_ANY, u"open video file", wx.DefaultPosition,
                                       wx.Size(90, 25), 0)
        gSizer1.Add(self.vedio_button2, 0, wx.ALL, 5)

        self.off_button3 = wx.Button(sbSizer2.GetStaticBox(), wx.ID_ANY, u"pause", wx.DefaultPosition, wx.Size(90, 25), 0)
        gSizer1.Add(self.off_button3, 0, wx.ALL, 5)
        sbSizer2.Add(gSizer1, 1, wx.EXPAND, 5)
        sbSizer1.Add(sbSizer2, 2, wx.EXPAND, 5)

        self.information_button1 = wx.Button(sbSizer2.GetStaticBox(), wx.ID_ANY, u"upload ",
                                             wx.DefaultPosition,
                                             wx.Size(90, 25), 0)
        gSizer1.Add(self.information_button1, 0, wx.ALL, 5)

        self.information_button2 = wx.Button(sbSizer2.GetStaticBox(), wx.ID_ANY, u"update ",
                                             wx.DefaultPosition,
                                             wx.Size(90, 25), 0)
        gSizer1.Add(self.information_button2, 0, wx.ALL, 5)

        sbSizer3 = wx.StaticBoxSizer(wx.StaticBox(sbSizer1.GetStaticBox(), wx.ID_ANY, u"fatigue detecting"), wx.VERTICAL)
        bSizer5 = wx.BoxSizer(wx.HORIZONTAL)
        self.yawn_checkBox1 = wx.CheckBox(sbSizer3.GetStaticBox(), wx.ID_ANY, u"yawning detecting", wx.Point(-1, -1),
                                          wx.Size(-1, 15), 0)
        self.yawn_checkBox1.SetValue(True)
        bSizer5.Add(self.yawn_checkBox1, 0, wx.ALL, 5)
        self.blink_checkBox2 = wx.CheckBox(sbSizer3.GetStaticBox(), wx.ID_ANY, u"blinking detecting", wx.Point(-1, -1),
                                           wx.Size(-1, 15), 0)
        self.blink_checkBox2.SetValue(True)
        bSizer5.Add(self.blink_checkBox2, 0, wx.ALL, 5)
        sbSizer3.Add(bSizer5, 1, wx.EXPAND, 5)
        bSizer6 = wx.BoxSizer(wx.HORIZONTAL)
        self.nod_checkBox7 = wx.CheckBox(sbSizer3.GetStaticBox(), wx.ID_ANY, u"nodding detecting", wx.Point(-1, -1), wx.Size(-1, 15),
                                         0)
        self.nod_checkBox7.SetValue(True)
        bSizer6.Add(self.nod_checkBox7, 0, wx.ALL, 5)
        self.m_staticText1 = wx.StaticText(sbSizer3.GetStaticBox(), wx.ID_ANY, u"testing interval(s):", wx.DefaultPosition,
                                           wx.Size(-1, 15), 0)
        self.m_staticText1.Wrap(-1)
        bSizer6.Add(self.m_staticText1, 0, wx.ALL, 5)
        m_listBox2Choices = [u"3", u"4", u"5", u"6", u"7", u"8"]
        self.m_listBox2 = wx.ListBox(sbSizer3.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.Size(50, 24),
                                     m_listBox2Choices, 0)
        bSizer6.Add(self.m_listBox2, 0, 0, 5)
        sbSizer3.Add(bSizer6, 1, wx.EXPAND, 5)
        sbSizer1.Add(sbSizer3, 2, 0, 5)
        sbSizer4 = wx.StaticBoxSizer(wx.StaticBox(sbSizer1.GetStaticBox(), wx.ID_ANY, u"absences detecting"), wx.VERTICAL)
        bSizer8 = wx.BoxSizer(wx.HORIZONTAL)
        self.m_checkBox4 = wx.CheckBox(sbSizer4.GetStaticBox(), wx.ID_ANY, u"absences detecting", wx.DefaultPosition, wx.Size(-1, 15),
                                       0)
        self.m_checkBox4.SetValue(True)
        bSizer8.Add(self.m_checkBox4, 0, wx.ALL, 5)
        self.m_staticText2 = wx.StaticText(sbSizer4.GetStaticBox(), wx.ID_ANY, u"absences interval(s):", wx.DefaultPosition,
                                           wx.Size(-1, 15), 0)
        self.m_staticText2.Wrap(-1)
        bSizer8.Add(self.m_staticText2, 0, wx.ALL, 5)
        m_listBox21Choices = [u"5", u"10", u"15", u"20", u"25", u"30"]
        self.m_listBox21 = wx.ListBox(sbSizer4.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.Size(50, 24),
                                      m_listBox21Choices, 0)
        bSizer8.Add(self.m_listBox21, 0, 0, 5)
        sbSizer4.Add(bSizer8, 1, 0, 5)
        sbSizer1.Add(sbSizer4, 1, 0, 5)
        #sbSizer5 = wx.StaticBoxSizer(wx.StaticBox(sbSizer1.GetStaticBox(), wx.ID_ANY, u"analysis area"), wx.VERTICAL)
        bSizer9 = wx.BoxSizer(wx.HORIZONTAL)
        #self.m_staticText3 = wx.StaticText(sbSizer5.GetStaticBox(), wx.ID_ANY, u"analysis area：   ", wx.DefaultPosition,
                                           #wx.DefaultSize, 0)
        #self.m_staticText3.Wrap(-1)
        #bSizer9.Add(self.m_staticText3, 0, wx.ALL, 5)
        #m_choice2Choices = [u"full screen", u"part of screen"]
        #self.m_choice2 = wx.Choice(sbSizer5.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize,
                                   #m_choice2Choices, 0)
        #self.m_choice2.SetSelection(0)
        #bSizer9.Add(self.m_choice2, 0, wx.ALL, 5)
        #sbSizer5.Add(bSizer9, 1, wx.EXPAND, 5)
        #sbSizer1.Add(sbSizer5, 1, 0, 5)
        sbSizer6 = wx.StaticBoxSizer(wx.StaticBox(sbSizer1.GetStaticBox(), wx.ID_ANY, u"status output"), wx.VERTICAL)
        self.m_textCtrl3 = wx.TextCtrl(sbSizer6.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition,
                                       wx.DefaultSize, wx.TE_MULTILINE | wx.TE_READONLY)
        sbSizer6.Add(self.m_textCtrl3, 1, wx.ALL | wx.EXPAND, 5)
        sbSizer1.Add(sbSizer6, 5, wx.EXPAND, 5)
        bSizer4.Add(sbSizer1, 1, wx.EXPAND, 5)
        bSizer2.Add(bSizer4, 3, wx.EXPAND, 5)
        bSizer1.Add(bSizer2, 1, wx.EXPAND, 5)

        self.SetSizer(bSizer1)
        self.Layout()
        self.Centre(wx.BOTH)

        # Connect Events
        self.m_choice1.Bind(wx.EVT_CHOICE, self.cameraid_choice)  # 绑定事件
        self.camera_button1.Bind(wx.EVT_BUTTON, self.camera_on)  # 开
        self.vedio_button2.Bind(wx.EVT_BUTTON, self.vedio_on)
        self.off_button3.Bind(wx.EVT_BUTTON, self.off)  # 关
        self.information_button1.Bind(wx.EVT_BUTTON, self.upload)
        self.information_button2.Bind(wx.EVT_BUTTON, self.update)

        self.m_listBox2.Bind(wx.EVT_LISTBOX, self.AR_CONSEC_FRAMES)  # 闪烁阈值设置
        self.m_listBox21.Bind(wx.EVT_LISTBOX, self.OUT_AR_CONSEC_FRAMES)  # 脱岗时间设置

        # 封面图片
        self.image_cover = wx.Image(COVER, wx.BITMAP_TYPE_ANY)
        # 显示图片在m_animCtrl1上
        self.bmp = wx.StaticBitmap(self.m_animCtrl1, -1, wx.Bitmap(self.image_cover))

        # 设置窗口标题的图标
        self.icon = wx.Icon('./images/123.ico', wx.BITMAP_TYPE_ICO)
        self.SetIcon(self.icon)
        # 系统事件
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        print("wxpython interface initialization is complete！")
        """参数"""
        # 默认为摄像头0
        self.VIDEO_STREAM = 0
        self.CAMERA_STYLE = False  # False未打开摄像头，True摄像头已打开
        # 闪烁阈值（秒）
        self.AR_CONSEC_FRAMES_check = 3
        self.OUT_AR_CONSEC_FRAMES_check = 5
        # 眼睛长宽比
        self.EYE_AR_THRESH = 0.2
        self.EYE_AR_CONSEC_FRAMES = self.AR_CONSEC_FRAMES_check
        # 打哈欠长宽比
        self.MAR_THRESH = 0.5
        self.MOUTH_AR_CONSEC_FRAMES = self.AR_CONSEC_FRAMES_check
        # 瞌睡点头
        self.HAR_THRESH = 0.3
        self.NOD_AR_CONSEC_FRAMES = self.AR_CONSEC_FRAMES_check

        """计数"""
        # 初始化帧计数器和眨眼总数
        self.COUNTER = 0
        self.TOTAL = 0
        # 初始化帧计数器和打哈欠总数
        self.mCOUNTER = 0
        self.mTOTAL = 0
        # 初始化帧计数器和点头总数
        self.hCOUNTER = 0
        self.hTOTAL = 0
        # 离职时间长度
        self.oCOUNTER = 0

        """姿态"""
        # 世界坐标系(UVW)：填写3D参考点，该模型参考http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
        self.object_pts = np.float32([[6.825897, 6.760612, 4.402142],  # 33左眉左上角
                                      [1.330353, 7.122144, 6.903745],  # 29左眉右角
                                      [-1.330353, 7.122144, 6.903745],  # 34右眉左角
                                      [-6.825897, 6.760612, 4.402142],  # 38右眉右上角
                                      [5.311432, 5.485328, 3.987654],  # 13左眼左上角
                                      [1.789930, 5.393625, 4.413414],  # 17左眼右上角
                                      [-1.789930, 5.393625, 4.413414],  # 25右眼左上角
                                      [-5.311432, 5.485328, 3.987654],  # 21右眼右上角
                                      [2.005628, 1.409845, 6.165652],  # 55鼻子左上角
                                      [-2.005628, 1.409845, 6.165652],  # 49鼻子右上角
                                      [2.774015, -2.080775, 5.048531],  # 43嘴左上角
                                      [-2.774015, -2.080775, 5.048531],  # 39嘴右上角
                                      [0.000000, -3.116408, 6.097667],  # 45嘴中央下角
                                      [0.000000, -7.415691, 4.070434]])  # 6下巴角

        # 相机坐标系(XYZ)：添加相机内参
        self.K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
                  0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
                  0.0, 0.0, 1.0]  # 等价于矩阵[fx, 0, cx; 0, fy, cy; 0, 0, 1]
        # 图像中心坐标系(uv)：相机畸变参数[k1, k2, p1, p2, k3]
        self.D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

        # 像素坐标系(xy)：填写凸轮的本征和畸变系数
        self.cam_matrix = np.array(self.K).reshape(3, 3).astype(np.float32)
        self.dist_coeffs = np.array(self.D).reshape(5, 1).astype(np.float32)

        # 重新投影3D点的世界坐标轴以验证结果姿势
        self.reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                                        [10.0, 10.0, -10.0],
                                        [10.0, -10.0, -10.0],
                                        [10.0, -10.0, 10.0],
                                        [-10.0, 10.0, 10.0],
                                        [-10.0, 10.0, -10.0],
                                        [-10.0, -10.0, -10.0],
                                        [-10.0, -10.0, 10.0]])
        # 绘制正方体12轴
        self.line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                           [4, 5], [5, 6], [6, 7], [7, 4],
                           [0, 4], [1, 5], [2, 6], [3, 7]]

    def __del__(self):
        pass

    def get_head_pose(self, shape):  # 头部姿态估计
        # （像素坐标集合）填写2D参考点，注释遵循https://ibug.doc.ic.ac.uk/resources/300-W/
        # 17左眉左上角/21左眉右角/22右眉左上角/26右眉右上角/36左眼左上角/39左眼右上角/42右眼左上角/
        # 45右眼右上角/31鼻子左上角/35鼻子右上角/48左上角/54嘴右上角/57嘴中央下角/8下巴角
        image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                                shape[39], shape[42], shape[45], shape[31], shape[35],
                                shape[48], shape[54], shape[57], shape[8]])
        # solvePnP计算姿势——求解旋转和平移矩阵：
        # rotation_vec表示旋转矩阵，translation_vec表示平移矩阵，cam_matrix与K矩阵对应，dist_coeffs与D矩阵对应。
        _, rotation_vec, translation_vec = cv2.solvePnP(self.object_pts, image_pts, self.cam_matrix, self.dist_coeffs)
        # projectPoints重新投影误差：原2d点和重投影2d点的距离（输入3d点、相机内参、相机畸变、r、t，输出重投影2d点）
        reprojectdst, _ = cv2.projectPoints(self.reprojectsrc, rotation_vec, translation_vec, self.cam_matrix,
                                            self.dist_coeffs)
        reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))  # 以8行2列显示

        # 计算欧拉角calc euler angle
        # 参考https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#decomposeprojectionmatrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)  # 罗德里格斯公式（将旋转矩阵转换为旋转向量）
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))  # 水平拼接，vconcat垂直拼接
        # decomposeProjectionMatrix将投影矩阵分解为旋转矩阵和相机矩阵
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

        pitch, yaw, roll = [math.radians(_) for _ in euler_angle]

        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))
        # print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))

        return reprojectdst, euler_angle  # 投影误差，欧拉角

    def eye_aspect_ratio(self, eye):
        # 垂直眼标志（X，Y）坐标
        A = dist.euclidean(eye[1], eye[5])  # 计算两个集合之间的欧式距离
        B = dist.euclidean(eye[2], eye[4])
        # 计算水平之间的欧几里得距离
        # 水平眼标志（X，Y）坐标
        C = dist.euclidean(eye[0], eye[3])
        # 眼睛长宽比的计算
        ear = (A + B) / (2.0 * C)
        # 返回眼睛的长宽比
        return ear

    def mouth_aspect_ratio(self, mouth):  # 嘴部
        A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
        B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
        C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
        mar = (A + B) / (2.0 * C)
        return mar

    # 处理存放所有人脸特征的 csv
    path_features_known_csv = "G:/pycharm project/python project/face detecting/1111.csv"
    csv_rd = pd.read_csv(path_features_known_csv, header=None)


    # 读取已知人脸数据
    # print known faces
    for i in range(csv_rd.shape[0]):
        features_someone_arr = []
        for j in range(0, len(csv_rd.iloc[i, :])):
            features_someone_arr.append(csv_rd.iloc[i, :][j])
        features_known_arr.append(features_someone_arr)
    print("Faces in Database：", len(features_known_arr))

    def _learning_face(self, event):
        """dlib的初始化调用"""
        # 使用人脸检测器get_frontal_face_detector
        self.detector = dlib.get_frontal_face_detector()
        # dlib的68点模型，使用作者训练好的特征预测器
        self.predictor = dlib.shape_predictor(
            "G:/pycharm project/python project/face detecting/model/shape_predictor_68_face_landmarks.dat")
        self.m_textCtrl3.AppendText(u"Loading model successfully!!\n")
        # 分别获取左右眼面部标志的索引
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

        # 建cv2摄像头对象，这里使用电脑自带摄像头，如果接了外部摄像头，则自动切换到外部摄像头
        self.cap = cv2.VideoCapture(self.VIDEO_STREAM)

        if self.cap.isOpened() == True:  # 返回true/false 检查初始化是否成功
            self.CAMERA_STYLE = True
            self.m_textCtrl3.AppendText(u"Open the camera successfully!!\n")
            time_start = time.time()
        else:
            self.m_textCtrl3.AppendText(u"Fail to open the camera!!\n")
            # 显示封面图
            self.bmp.SetBitmap(wx.Bitmap(self.image_cover))
        # 成功打开视频，循环读取视频流
        while (self.cap.isOpened()):
            # cap.read()
            # 返回两个值：
            #    一个布尔值true/false，用来判断读取视频是否成功/是否到视频末尾
            #    图像对象，图像的三维矩阵
            flag, im_rd = self.cap.read()
            kk = cv2.waitKey(1)
            # 取灰度
            img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)

            # 使用人脸检测器检测每一帧图像中的人脸。并返回人脸数faces
            faces = self.detector(img_gray, 0)

            # 待会要写的字体 font to write later
            font = cv2.FONT_HERSHEY_COMPLEX

            # 存储当前摄像头中捕获到的所有人脸的坐标/名字
            # the list to save the positions and names of current faces captured
            pos_namelist = []
            name_namelist = []

            # 计算两个128D向量间的欧式距离
            # compute the e-distance between two 128D features
            def return_euclidean_distance(feature_1, feature_2):
                feature_1 = np.array(feature_1)
                feature_2 = np.array(feature_2)
                dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
                return dist
            # 如果检测到人脸
            if (len(faces) != 0):
                # enumerate方法同时返回数据对象的索引和数据，k为索引，d为faces中的对象
                features_cap_arr = []
                for k, d in enumerate(faces):
                    # 用红色矩形框出人脸
                    cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 1)
                    # 使用预测器得到68点数据的坐标
                    shape = self.predictor(im_rd, d)
                    features_cap_arr.append(facerec.compute_face_descriptor(im_rd, shape))
                    # 圆圈显示每个特征点
                    for i in range(68):
                        cv2.circle(im_rd, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1, 8)
                    for k in range(len(faces)):
                        print("##### camera person", k + 1, "#####")
                        # 让人名跟随在矩形框的下方
                        # 确定人名的位置坐标
                        # 先默认所有人不认识，是 unknown
                        # set the default names of faces with "unknown"
                        name_namelist.append("unknown")

                        # 每个捕获人脸的名字坐标 the positions of faces captured
                        pos_namelist.append(
                            tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                        # 对于某张人脸，遍历所有存储的人脸特征
                        # for every faces detected, compare the faces in the database
                        e_distance_list = []
                        for i in range(len(features_known_arr)):
                            # 如果 person_X 数据不为空
                            if str(features_known_arr[i][0]) != '0.0':
                                print("with person", str(i + 1), "the e distance: ", end='')
                                e_distance_tmp = return_euclidean_distance(features_cap_arr[k], features_known_arr[i])
                                print(e_distance_tmp)
                                e_distance_list.append(e_distance_tmp)
                            else:
                                # 空数据 person_X
                                e_distance_list.append(999999999)
                        # 找出最接近的一个人脸数据是第几个
                        # Find the one with minimum e distance
                        similar_person_num = e_distance_list.index(min(e_distance_list))
                        print("Minimum e distance with person", int(similar_person_num) + 1)

                        # 计算人脸识别特征与数据集特征的欧氏距离
                        # 距离小于0.4则标出为可识别人物
                        if min(e_distance_list) < 0.4:
                            # 这里可以修改摄像头中标出的人名
                            # Here you can modify the names shown on the camera
                            # 1、遍历文件夹目录
                            folder_name = 'G:/pycharm project/python project/face detecting/pictures/people'
                            # 最接近的人脸
                            sum = similar_person_num + 1
                            key_id = 1  # 从第一个人脸数据文件夹进行对比
                            # 获取文件夹中的文件名:1wang、2zhou、3...
                            file_names = os.listdir(folder_name)
                            for name in file_names:
                                # print(name+'->'+str(key_id))
                                if sum == key_id:
                                    # winsound.Beep(300,500)# 响铃：300频率，500持续时间
                                    name_namelist[k] = name[1:]  # 人名删去第一个数字（用于视频输出标识）
                                key_id += 1
                            # 播放欢迎光临音效
                            # playsound('D:/myworkspace/JupyterNotebook/People/music/welcome.wav')
                            # print("May be person "+str(int(similar_person_num)+1))
                            # -----------筛选出人脸并保存到visitor文件夹------------
                            for i, d in enumerate(faces):
                                x1 = d.top() if d.top() > 0 else 0
                                y1 = d.bottom() if d.bottom() > 0 else 0
                                x2 = d.left() if d.left() > 0 else 0
                                y2 = d.right() if d.right() > 0 else 0
                                face = im_rd[x1:y1, x2:y2]
                                size = 64
                                face = cv2.resize(face, (size, size))
                                # 要存储visitor人脸图像文件的路径
                                # path_visitors_save_dir = "D:/myworkspace/JupyterNotebook/People/visitor/known"
                                path_visitors_save_dir = "G:/pycharm project/python project/face detecting/pictures/people/visitors/known"
                                # 存储格式：2019-06-24-14-33-40wang.jpg
                                now_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                                save_name = str(now_time) + str(name_namelist[k]) + '.jpg'
                                # print(save_name)
                                # 本次图片保存的完整url
                                save_path = path_visitors_save_dir + '/' + save_name
                                # 遍历visitor文件夹所有文件名
                                visitor_names = os.listdir(path_visitors_save_dir)
                                visitor_name = ''
                                for name in visitor_names:
                                    # 名字切片到分钟数：2019-06-26-11-33-00wangyu.jpg
                                    visitor_name = (name[0:16] + '-00' + name[19:])
                                # print(visitor_name)
                                visitor_save = (save_name[0:16] + '-00' + save_name[19:])
                                # print(visitor_save)
                                # 一分钟之内重复的人名不保存
                                if visitor_save != visitor_name:
                                    cv2.imwrite(save_path, face)
                                    print(
                                        '新存储：' + path_visitors_save_dir + '/' + str(now_time) + str(
                                            name_namelist[k]) + '.jpg')
                                else:
                                    print('重复，未保存！')

                        else:
                            # 播放无法识别音效
                            # playsound('D:/myworkspace/JupyterNotebook/People/music/sorry.wav')
                            print("Unknown person")
                            # -----保存图片-------
                            # -----------筛选出人脸并保存到visitor文件夹------------
                            for i, d in enumerate(faces):
                                x1 = d.top() if d.top() > 0 else 0
                                y1 = d.bottom() if d.bottom() > 0 else 0
                                x2 = d.left() if d.left() > 0 else 0
                                y2 = d.right() if d.right() > 0 else 0
                                face = im_rd[x1:y1, x2:y2]
                                size = 64
                                face = cv2.resize(face, (size, size))
                                # 要存储visitor-》unknown人脸图像文件的路径
                                path_visitors_save_dir = "G:/pycharm project/python project/face detecting/pictures/people/visitors/unknown"
                                # 存储格式：2019-06-24-14-33-40unknown.jpg
                                now_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                                # print(save_name)
                                # 本次图片保存的完整url
                                save_path = path_visitors_save_dir + '/' + str(now_time) + 'unknown.jpg'
                                cv2.imwrite(save_path, face)
                                print('新存储：' + path_visitors_save_dir + '/' + str(now_time) + 'unknown.jpg')
                    # 在人脸框下面写人脸名字
                    # write names under rectangle
                    for i in range(len(faces)):
                        cv2.putText(im_rd, name_namelist[i], pos_namelist[i], font, 0.8, (0, 255, 255), 1,cv2.LINE_AA)

                    print("Faces in camera now:", name_namelist, "\n")
                # 将脸部特征信息转换为数组array的格式
                    shape = face_utils.shape_to_np(shape)
                    """
                    打哈欠
                    """
                    if self.yawn_checkBox1.GetValue() == True:
                        # 嘴巴坐标
                        mouth = shape[mStart:mEnd]
                        # 打哈欠
                        mar = self.mouth_aspect_ratio(mouth)
                        # 使用cv2.convexHull获得凸包位置，使用drawContours画出轮廓位置进行画图操作
                        mouthHull = cv2.convexHull(mouth)
                        cv2.drawContours(im_rd, [mouthHull], -1, (0, 255, 0), 1)
                        # 同理，判断是否打哈欠
                        if mar > self.MAR_THRESH:  # 张嘴阈值0.5
                            self.mCOUNTER += 1
                        else:
                            # 如果连续3次都小于阈值，则表示打了一次哈欠
                            if self.mCOUNTER >= self.MOUTH_AR_CONSEC_FRAMES:  # 阈值：3
                                self.mTOTAL += 1
                                # 显示
                                cv2.putText(im_rd, "Yawning!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                self.m_textCtrl3.AppendText(
                                    time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + u"yawn!!!\n")
                            # 重置嘴帧计数器
                            self.mCOUNTER = 0
                        cv2.putText(im_rd, "COUNTER: {}".format(self.mCOUNTER), (150, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 255), 2)
                        cv2.putText(im_rd, "MAR: {:.2f}".format(mar), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2)
                        cv2.putText(im_rd, "Yawning: {}".format(self.mTOTAL), (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (255, 255, 0), 2)
                    else:
                        pass
                    """
                    眨眼
                    """
                    if self.blink_checkBox2.GetValue() == True:
                        # 提取左眼和右眼坐标
                        leftEye = shape[lStart:lEnd]
                        rightEye = shape[rStart:rEnd]
                        # 构造函数计算左右眼的EAR值，使用平均值作为最终的EAR
                        leftEAR = self.eye_aspect_ratio(leftEye)
                        rightEAR = self.eye_aspect_ratio(rightEye)
                        ear = (leftEAR + rightEAR) / 2.0
                        leftEyeHull = cv2.convexHull(leftEye)
                        rightEyeHull = cv2.convexHull(rightEye)
                        # 使用cv2.convexHull获得凸包位置，使用drawContours画出轮廓位置进行画图操作
                        cv2.drawContours(im_rd, [leftEyeHull], -1, (0, 255, 0), 1)
                        cv2.drawContours(im_rd, [rightEyeHull], -1, (0, 255, 0), 1)
                        # 循环，满足条件的，眨眼次数+1
                        if ear < self.EYE_AR_THRESH:  # 眼睛长宽比：0.2
                            self.COUNTER += 1

                        else:
                            # 如果连续3次都小于阈值，则表示进行了一次眨眼活动
                            if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:  # 阈值：3
                                self.TOTAL += 1
                                self.m_textCtrl3.AppendText(
                                    time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + u"blink!!!\n")
                            # 重置眼帧计数器
                            self.COUNTER = 0
                        # 第十四步：进行画图操作，同时使用cv2.putText将眨眼次数进行显示
                        cv2.putText(im_rd, "Faces: {}".format(len(faces)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2)
                        cv2.putText(im_rd, "COUNTER: {}".format(self.COUNTER), (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2)
                        cv2.putText(im_rd, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2)
                        cv2.putText(im_rd, "Blinks: {}".format(self.TOTAL), (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (255, 255, 0), 2)
                    else:
                        pass
                    """
                    瞌睡点头
                    """
                    if self.nod_checkBox7.GetValue() == True:
                        # 获取头部姿态
                        reprojectdst, euler_angle = self.get_head_pose(shape)
                        har = euler_angle[0, 0]  # 取pitch旋转角度
                        if har > self.HAR_THRESH:  # 点头阈值0.3
                            self.hCOUNTER += 1
                        else:
                            # 如果连续3次都小于阈值，则表示瞌睡点头一次
                            if self.hCOUNTER >= self.NOD_AR_CONSEC_FRAMES:  # 阈值：3
                                self.hTOTAL += 1
                                self.m_textCtrl3.AppendText(
                                    time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + u"sleepy nod!!! \n")
                            # 重置点头帧计数器
                            self.hCOUNTER = 0
                        # 绘制正方体12轴(视频流尺寸过大时，reprojectdst会超出int范围，建议压缩检测视频尺寸)
                        # for start, end in self.line_pairs:
                        # im_rd = im_rd.astype(int)
                        # print(reprojectdst)[start]
                        # cv2.line(im_rd, reprojectdst[start], reprojectdst[end], (0, 0, 255))
                        # 显示角度结果
                        cv2.putText(im_rd, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), thickness=2)  # GREEN
                        cv2.putText(im_rd, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (150, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)  # BLUE
                        cv2.putText(im_rd, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (300, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), thickness=2)  # RED
                        cv2.putText(im_rd, "Nod: {}".format(self.hTOTAL), (450, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (255, 255, 0), 2)
                    else:
                        pass

                print('Real-time mouth aspect ratio:{:.2f} '.format(mar) + "\tYawn or not：" + str([False, True][mar > self.MAR_THRESH]))
                print('Real-time eye aspect ratio:{:.2f} '.format(ear) + "\tBlink or not:" + str([False, True][self.COUNTER >= 1]))
            else:
                # 没有检测到人脸
                self.oCOUNTER += 1
                cv2.putText(im_rd, "No Face", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                if self.oCOUNTER >= self.OUT_AR_CONSEC_FRAMES_check:
                    self.m_textCtrl3.AppendText(time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + u"absence!!!\n")
                    self.oCOUNTER = 0

            # 确定疲劳提示:眨眼50次，打哈欠15次，瞌睡点头30次
            time_end = time.time()
            timecost = time_end - time_start
            print(timecost)
            if ((self.TOTAL >= 50 or self.mTOTAL >= 15 or self.hTOTAL >= 30) and (timecost < 200)) or (timecost>14400):
                cv2.putText(im_rd, "SLEEP!!!", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

                # self.m_textCtrl3.AppendText(u"疲劳")

            # opencv中imread的图片内部是BGR排序，wxPython的StaticBitmap需要的图片是RGB排序，不转换会出现颜色变换
            height, width = im_rd.shape[:2]
            image1 = cv2.cvtColor(im_rd, cv2.COLOR_BGR2RGB)
            pic = wx.Bitmap.FromBuffer(width, height, image1)
            # 显示图片在panel上：
            self.bmp.SetBitmap(pic)

        # 释放摄像头
        self.cap.release()

    def update(self,event):
        people = os.listdir(path_images_from_camera)
        people.sort()

        # with open("D:/myworkspace/JupyterNotebook/People/feature/features2_all.csv", "w", newline="") as csvfile:
        with open("G:/pycharm project/python project/face detecting/1111.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for person in people:
                print("##### " + person + " #####")
                # Get the mean/average features of face/personX, it will be a list with a length of 128D
                features_mean_personX = return_features_mean_personX(path_images_from_camera + person)
                writer.writerow(features_mean_personX)
                print("特征均值 / The mean of features:", list(features_mean_personX))
                print('\n')
            # print("所有录入人脸数据存入 / Save all the features of faces registered into: D:/myworkspace/JupyterNotebook/People/feature/features_all2.csv")
            print(
                "所有录入人脸数据存入 / Save all the features of faces registered into: G:/pycharm project/python project/face detecting/1111.csv")

            # import _thread
            # # 创建子线程，按钮调用这个方法，
            # _thread.start_new_thread(self._learning_face, (event,))
    def upload(self,event):

        dlg = wx.MessageDialog(None, u'Is this your first time using it？', u'Operating hints',wx.YES_NO | wx.ICON_QUESTION)
        if (dlg.ShowModal() == wx.ID_YES):
            # dlg = wx.TextEntryDialog()
            # dlg.Destroy()
            # dlg = wx.TextEntryDialog(None,u'Please create a folder named "number"+"letter", for instance:"1czx')
            folder_name = input('Please create a folder named "number"+"letter", for instance:"1czx\n')
            path = r'G:\pycharm project\python project\face detecting\pictures\people' + '/' +folder_name
            if not os.path.exists(path):
                os.makedirs(path)
                print('create folder successfully')
            else:
                print('fail to create the folder,it is exist')
                dlg.Destroy()  # 取消弹窗
        else:
            dialog = wx.FileDialog(self, u"choose photos", os.getcwd(), '', wildcard="(*.jpg)|*.jpg",
                                   style=wx.FD_OPEN | wx.FD_CHANGE_DIR)
            if dialog.ShowModal() == wx.ID_OK:
                path = str(dialog.GetPath())  # 更新全局变量路径
                filepath = os.path.basename(path)
                # print(path,filepath)
                folder_name = input('Please choose your folder named "number"+"letter", for instance:"1czx\n')
                path1 = r'G:\pycharm project\python project\face detecting\pictures\people' + '/' + folder_name + '/' + filepath
                if not os.path.exists(path1):
                    shutil.copyfile(dialog.GetPath(), path1)
                    print('upload photo successfully')
                else:
                    print('fail to create the folder,it is exist')


                #os.makedirs(path)
            # 选择文件夹对话框窗口
        # dialog = wx.FileDialog(self, u"choose videos", os.getcwd(), '', wildcard="(*.mp4)|*.mp4",
        #                            style=wx.FD_OPEN | wx.FD_CHANGE_DIR)
        # if dialog.ShowModal() == wx.ID_OK:
        #         # 如果确定了选择的文件夹，将文件夹路径写到m_textCtrl3控件
        #         # self.m_textCtrl3.SetValue(u"文件路径:" + dialog.GetPath() + "\n")
        #         # self.VIDEO_STREAM = str(dialog.GetPath())  # 更新全局变量路径

                # dialog.Destroy
        """使用多线程，子线程运行后台的程序，主线程更新前台的UI，这样不会互相影响"""
        # import _thread
        # # 创建子线程，按钮调用这个方法，
        # _thread.start_new_thread(self._learning_face, (event,))


    def camera_on(self, event):
        """使用多线程，子线程运行后台的程序，主线程更新前台的UI，这样不会互相影响"""
        import _thread
        # 创建子线程，按钮调用这个方法，
        _thread.start_new_thread(self._learning_face, (event,))

    def cameraid_choice(self, event):
        # 摄像头编号
        cameraid = int(event.GetString()[-1])  # 截取最后一个字符
        if cameraid == 0:
            self.m_textCtrl3.AppendText(u"Prepare to open the local camera!!!\n")
        if cameraid == 1 or cameraid == 2:
            self.m_textCtrl3.AppendText(u"Prepart to open the external camera!!!\n")
        self.VIDEO_STREAM = cameraid

    def vedio_on(self, event):
        if self.CAMERA_STYLE == True:  # 释放摄像头资源
            # 弹出关闭摄像头提示窗口
            dlg = wx.MessageDialog(None, u'Are you sure you want to close it？', u'Operating hints', wx.YES_NO | wx.ICON_QUESTION)
            if (dlg.ShowModal() == wx.ID_YES):
                self.cap.release()  # 释放摄像头
                self.bmp.SetBitmap(wx.Bitmap(self.image_cover))  # 封面
                dlg.Destroy()  # 取消弹窗
        # 选择文件夹对话框窗口
        dialog = wx.FileDialog(self, u"choose videos", os.getcwd(), '', wildcard="(*.mp4)|*.mp4",
                               style=wx.FD_OPEN | wx.FD_CHANGE_DIR)
        if dialog.ShowModal() == wx.ID_OK:
            # 如果确定了选择的文件夹，将文件夹路径写到m_textCtrl3控件
            self.m_textCtrl3.SetValue(u"文件路径:" + dialog.GetPath() + "\n")
            self.VIDEO_STREAM = str(dialog.GetPath())  # 更新全局变量路径
            dialog.Destroy
            """使用多线程，子线程运行后台的程序，主线程更新前台的UI，这样不会互相影响"""
            import _thread
            # 创建子线程，按钮调用这个方法，
            _thread.start_new_thread(self._learning_face, (event,))

    def AR_CONSEC_FRAMES(self, event):
        self.m_textCtrl3.AppendText(u"设置疲劳间隔为:\t" + event.GetString() + "秒\n")
        self.AR_CONSEC_FRAMES_check = int(event.GetString())

    def OUT_AR_CONSEC_FRAMES(self, event):
        self.m_textCtrl3.AppendText(u"设置脱岗间隔为:\t" + event.GetString() + "秒\n")
        self.OUT_AR_CONSEC_FRAMES_check = int(event.GetString())

    def off(self, event):
        """关闭摄像头，显示封面页"""
        self.cap.release()
        self.bmp.SetBitmap(wx.Bitmap(self.image_cover))

    def OnClose(self, evt):
        """关闭窗口事件函数"""
        dlg = wx.MessageDialog(None, u'Are you sure you want to close it？', u'Operating hints', wx.YES_NO | wx.ICON_QUESTION)
        if (dlg.ShowModal() == wx.ID_YES):
            self.Destroy()
        print("detecting finish")


class main_app(wx.App):
    """
     在OnInit() 里边申请Frame类，这样能保证一定是在app后调用，
     这个函数是app执行完自己的__init__函数后就会执行
    """

    # OnInit 方法在主事件循环开始前被wxPython系统调用，是wxpython独有的
    def OnInit(self):
        self.frame = Fatigue_detecting(parent=None, title="Fatigue Demo")
        self.frame.Show(True)
        return True


if __name__ == "__main__":
    app = main_app()
    app.MainLoop()

