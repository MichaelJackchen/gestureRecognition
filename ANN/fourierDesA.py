import math
import cv2
import numpy as np

def getHandFeatures(binariedImage):
    '''
    基于傅里叶描绘子提取手部轮廓特征
    :return:
    '''
    # opencv高版本返回两个值
    contours, hierarchy = cv2.findContours(
        binariedImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # 寻找最大轮廓
    maxSize = 0
    for i in range(0,len(contours)):
        if len(contours[i]) > maxSize:
            maxSize = len(contours[i])
            contourNum = i
    # 创建一个白色背景板，在上面画上最大轮廓
    # whiteBack = np.zeros((binariedImage.shape[0],binariedImage.shape[1],3),dtype=np.uint8)
    # whiteBack[0:binariedImage.shape[0]-1, 0:binariedImage.shape[1]-1] = 255
    # 第三个参数为最大轮廓的索引
    # cv2.drawContours(whiteBack, contours, contourNum, (0,0,0), 1)

    # 计算图像的傅里叶描绘子,存储傅里叶变换后的系数(前14位)
    f = []
    fd = []
    for i in range(0,maxSize):
        sumx = sumy = 0.0
        for j in range(0,maxSize):
            p = contours[contourNum][j][0]
            x= p[0]
            y = p[1]
            sumx += (x * math.cos(2 * math.pi * i * j / maxSize) + y * math.sin(2 * math.pi * i * j / maxSize));
            sumy += (y * math.cos(2 * math.pi * i * j / maxSize) - x * math.sin(2 * math.pi * i * j / maxSize));
        f.append(math.sqrt((sumx * sumx) + (sumy * sumy)))
    fd.append(0.0)
    # 进行归一化，然后放入最终结果中
    for k in range(2,16):
        f[k] = f[k] / f[1];
        fd.append(f[k]);
    # 输出最终的手势特征
    out = np.zeros(len(fd),dtype=np.float32)
    for i in range(0,len(fd)):
        out[i] = fd[i]
    return out