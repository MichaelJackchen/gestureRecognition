import cv2
import numpy as np
import math


def imageIn():
    """
    传递输入的手部图片
    :return:输出图片像素矩阵信息
    """
    # 路径前有数字要加上双斜杠
    image = cv2.imread("images\\IMG_1129.jpg")
    # image_gray = Gray_img(image)
    biImage = skinMask2(image)
    lunkuo = getHandFeatures(biImage)
    # erzhi = skinMask(image)
    cv2.imshow("image", lunkuo)
    cv2.waitKey(0)

def Gray_img(img):
    '''
    图片灰度化
    :param img:
    :return:
    '''
    # img_gray = cv.cvtColor(img_2, cv.COLOR_RGB2GRAY)
    (h, w, c) = img.shape
    img_b = img[:, :, 0]
    img_g = img[:, :, 1]
    img_r = img[:, :, 2]
    img_gray = img_r * 0.299 + img_g * 0.587 + img_b * 0.114
    img_gray = img_gray.astype(np.uint8)  # (1)
    return img_gray

def skinMask(images):
    """
    将手部图片二值化,基于简单RGB阈值判断
    判别式:R>95 && G>40 && B>20 && R>G && R>B && Max(R,G,B)-Min(R,G,B)>15 && Abs(R-G)>15
    :param images:
    :return:二值化后的图片
    """
    for i in range(images.shape[0]):
        for j in range(images.shape[1]):
            if images[i][j][2] > 95 and images[i][j][1] > 40 and images[i][j][0] > 20 \
                    and images[i][j][2] > images[i][j][1] and images[i][j][2] > images[i][j][0] \
                    and max(images[i][j][2], images[i][j][1], images[i][j][0]) - \
                    min(images[i][j][2], images[i][j][1], images[i][j][0]) > 15 \
                    and abs(images[i][j][2] - images[i][j][1]) > 15:
                images[i][j][0] = images[i][j][1] = images [i][j][2] = 255

    return images

def skinMask2(images):
    '''
    基于椭圆皮肤模型,YCrCb:Y-明亮度,(cr,cb)-色度
    :param images:
    :return:二值化后的图片:
    '''
    #生成椭圆模型
    skinCrCbHist = np.zeros((256,256))
    #新建单通道的灰度图,不指定类型默认为float32(提取轮廓时会报错),图像需要无符号整数即uint8
    newImage = np.zeros((images.shape[0],images.shape[1],1),dtype=np.uint8)
    # 轴(以及中心)必须是整数元组,而不是浮点数
    oval = cv2.ellipse(skinCrCbHist, (113,155), (23,15), 43.0, 0.0, 360.0, (255, 255, 255), -1)
    # 将图片转换为YCrCb色彩空间的图片
    imageY = cv2.cvtColor(images, cv2.COLOR_BGR2YCrCb)
    # 分离Y-CR-CB参数
    imageY_Cr = imageY[:,:,1]
    imageY_Cb = imageY[:,:,2]
    for i in range(images.shape[0]):
        for j in range(images.shape[1]):
            if skinCrCbHist[imageY_Cr[i][j]][imageY_Cb[i][j]] > 0:
                newImage[i][j] = 255
    return newImage

# def skinMask3(roi):
#     skinCrCbHist = np.zeros((256,256), dtype= np.uint8)
#     cv2.ellipse(skinCrCbHist, (113,155),(23,25), 43, 0, 360, (255,255,255), -1) #绘制椭圆弧线
#     YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB) #转换至YCrCb空间
#     (y,Cr,Cb) = cv2.split(YCrCb) #拆分出Y,Cr,Cb值
#     skin = np.zeros(Cr.shape, dtype = np.uint8) #掩膜
#     (x,y) = Cr.shape
#     for i in range(0, x):
#         for j in range(0, y):
#             if skinCrCbHist [Cr[i][j], Cb[i][j]] > 0: #若不在椭圆区间中
#                 skin[i][j] = 255
#     res = cv2.bitwise_and(roi,roi, mask = skin)
#     return res

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
    whiteBack = np.zeros((binariedImage.shape[0],binariedImage.shape[1],3),dtype=np.uint8)
    whiteBack[0:binariedImage.shape[0]-1, 0:binariedImage.shape[1]-1] = 255
    # 第三个参数为最大轮廓的索引
    cv2.drawContours(whiteBack, contours, contourNum, (0,0,0), 1)

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
    return whiteBack

def getAllFeatures():
    '''
    提取所有训练样本的特征，按照特征值与标签矩阵形式存储
    :return:
    '''

imageIn()
