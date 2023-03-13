import cv2
import numpy as np


def imageIn():
    """
    传递输入的手部图片
    :return:输出图片像素矩阵信息
    """
    # 路径前有数字要加上双斜杠
    image = cv2.imread("images\\1_30.png")
    # print("Blue:",image[50][50][0])
    # print(type(image.shape),image.shape[1])
    skinMask2(image)
    # erzhi = skinMask(image)
    # cv2.imshow("image", erzhi)
    # cv2.waitKey(0)


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
    skinCrCbHist = np.zeros((256,256,1))
    #新建单通道的灰度图
    newImage = np.zeros((images.shape[0],images.shape[1],1))
    # 轴(以及中心)必须是整数元组,而不是浮点数
    oval = cv2.ellipse(skinCrCbHist, (113,155), (23,15), 43.0, 0.0, 360.0, (255, 255, 255), -1)
    # 将图片转换为YCrCb色彩空间的图片
    imageY = cv2.cvtColor(images, cv2.COLOR_BGR2YCrCb)
    # 分离Y-CR-CB参数
    imageY_Cr = imageY[:,:,1]
    imageY_Cb = imageY[:,:,2]
    for i in range(images.shape[0]):
        for j in range(images.shape[1]):
            if oval[imageY_Cr[i][j]][imageY_Cb[i][j]] > 0:
                newImage[i][j] = 255
    return newImage

imageIn()

