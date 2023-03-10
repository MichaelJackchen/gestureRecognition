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
    erzhi = shinMask(image)
    cv2.imshow("image", erzhi)
    cv2.waitKey(0)


def shinMask(images):
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

imageIn()
