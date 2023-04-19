import cv2
import numpy as np
import math
import mediapipe as mp
import fourierDescriptor as fd


def imageIn():
    """
    传递输入的手部图片
    :return:输出图片像素矩阵信息
    """
    # 路径前有数字要加上双斜杠
    filename = "images\\test\\1_30.png"
    image = cv2.imread(filename)
    # image_gray = Gray_img(image)
    # x,y = bonePoints(filename)    后面用
    # 2_1可以用来实时检测
    # biImage = skinMask2_1(image,x,y)
    biImage = skinMask2_1(image)
    # lunkuo = getHandFeatures(biImage)
    # erzhi = skinMask(image)
    cv2.imshow("image", biImage)
    ret, fourier_result = fd.fourierDesciptor(biImage)
    cv2.imshow("image2", ret)
    '''
    kernel = np.ones((3, 3), np.uint8)  # 设置卷积核
    erosion = cv2.erode(biImage, kernel)  # 腐蚀操作
    cv2.imshow("erosion", erosion)
    dilation = cv2.dilate(erosion, kernel)  # 膨胀操作
    cv2.imshow("dilation", dilation)
    '''
    # 形态学处理
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
    newImage = np.zeros((images.shape[0],images.shape[1],1),dtype=np.uint8)
    for i in range(images.shape[0]):
        for j in range(images.shape[1]):
            if images[i][j][2] > 95 and images[i][j][1] > 40 and images[i][j][0] > 20 \
                    and images[i][j][2] > images[i][j][1] and images[i][j][2] > images[i][j][0] \
                    and max(images[i][j][2], images[i][j][1], images[i][j][0]) - \
                    min(images[i][j][2], images[i][j][1], images[i][j][0]) > 15 \
                    and abs(images[i][j][2] - images[i][j][1]) > 15:
                # images[i][j][0] = images[i][j][1] = images [i][j][2] = 255
                newImage[i][j] = 255

    return newImage


def skinMask2(images):
    '''
    基于椭圆皮肤模型,YCrCb:Y-明亮度,(cr,cb)-色度,x,y为手掌中心点坐标
    :param images:
    :return:二值化后的图片:
    '''
    #生成椭圆模型
    skinCrCbHist = np.zeros((256,256))
    #新建单通道的灰度图,不指定类型默认为float32(提取轮廓时会报错),图像需要无符号整数即uint8
    newImage = np.zeros((images.shape[0],images.shape[1],1),dtype=np.uint8)
    # 轴(以及中心)必须是整数元组,而不是浮点数
    oval = cv2.ellipse(skinCrCbHist, (113,155), (23,25), 43.0, 0.0, 360.0, (255, 255, 255), -1)
    # 将图片转换为YCrCb色彩空间的图片
    imageY = cv2.cvtColor(images, cv2.COLOR_BGR2YCrCb)
    # 分离Y-CR-CB参数
    imageY_Cr = imageY[:,:,1]
    imageY_Cb = imageY[:,:,2]
    for i in range(images.shape[0]):
        for j in range(images.shape[1]):
            # d = math.sqrt((i - x) ** 2 + (j - y) ** 2)
            if skinCrCbHist[imageY_Cr[i][j]][imageY_Cb[i][j]] > 0:
                newImage[i][j] = 255
    return newImage

# def skinMask2_1(images,x,y):
def skinMask2_1(images):

    '''
    改进后的椭圆皮肤模型,YCrCb:Y-明亮度,(cr,cb)-色度,x,y为手掌中心点坐标
    :param images:
    :return:二值化后的图片:
    '''
    #1.生成椭圆模型
    skinCrCbHist = np.zeros((256,256), dtype= np.uint8)
    # 绘制一个椭圆弧线  轴(以及中心)必须是整数元组,而不是浮点数
    cv2.ellipse(skinCrCbHist, (113, 155), (23, 25), 43.0, 0.0, 360.0, (255, 255, 255), -1)
    #2.将图片转换为YCrCb色彩空间的图片
    imageY = cv2.cvtColor(images, cv2.COLOR_BGR2YCR_CB)
    #3.分离Y-CR-CB参数
    (imageY_Y,imageY_Cr,imageY_Cb) = cv2.split(imageY)
    skin = np.zeros(imageY_Cr.shape, dtype=np.uint8)   #掩模,与原图相等大小的卷积核
    (crx,cry) = imageY_Cr.shape
    for i in range(0, crx):
        for j in range(0, cry):
            # d = math.sqrt((i - x) ** 2 + (j - y) ** 2)
            # if skinCrCbHist[imageY_Cr[i][j]][imageY_Cb[i][j]] > 0 and d<=80: 后面用,前期训练时可能检测不出静态图片里的手掌
            if skinCrCbHist[imageY_Cr[i][j]][imageY_Cb[i][j]] > 0:
                skin[i][j] = 255
    res = cv2.bitwise_and(images,images,mask = skin)    #mask: 要提取的区域,即在原图像中保留了处理后的掩膜区域（手部区域）
    return res

def skinMask3(images):
    '''
    YCrCb颜色空间的Cr分量+Otsu法阈值分割算法
    '''
    imageY = cv2.cvtColor(images, cv2.COLOR_BGR2YCR_CB)
    (imageY_Y, imageY_Cr, imageY_Cb) = cv2.split(imageY)
    cr = cv2.GaussianBlur(imageY_Cr, (5,5), 0)
    # otsu处理
    _, skin = cv2.threshold(cr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    res = cv2.bitwise_and(images, images, mask=skin)
    return res


def bonePoints(name):
    sample_img = cv2.imread(name)
    mp_drawing = mp.solutions.drawing_utils
    # 定义手部类和存储变量
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
            static_image_mode=True, #检测图片时设置为True,视频流设置为False
            max_num_hands=1,
            min_detection_confidence=0.75)
            # plt.figure(figsize=[10,10])
            # plt.title("sample Image");plt.axis('off');plt.imshow(sample_img[:,:,::-1]);plt.show()

    results = hands.process(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
    image_height, image_width, _ = sample_img.shape
    if results.multi_hand_landmarks:
        for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
        #     print(f'HAND NUMBER: {hand_no + 1}')
        #     print('-----------------------')
            # 共有21个坐标，这里显示了两个
            # print(f'{mp_hands.HandLandmark(9).name}:')
            handCenter_x = hand_landmarks.landmark[mp_hands.HandLandmark(9).value].x * image_width
            handCenter_y = hand_landmarks.landmark[mp_hands.HandLandmark(9).value].y * image_height
            print('x:',handCenter_x,"y:",handCenter_y)
            return handCenter_x,handCenter_y
    else:
        return 0,0

def getHandFeatures(binariedImage):
    '''
    基于傅里叶描绘子提取手部轮廓特征
    :return:
    '''
    # opencv高版本返回两个值

    gray = cv2.cvtColor(binariedImage, cv2.COLOR_BGR2GRAY)
    # 拉普拉斯算子提取轮廓点坐标
    dst = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    # 取像素绝对值
    Laplacian = cv2.convertScaleAbs(dst)
    # findcontours接受的参数为二值图即黑白图
    contours, hierarchy = cv2.findContours(
        Laplacian, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
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
    # cv2.imshow("bianyuan",whiteBack)
    # 计算图像的傅里叶描绘子,存储傅里叶变换后的系数(前15位)
    f = []
    fd = []
    for i in range(0,maxSize):
        sumx = sumy = 0.0
        for j in range(0,maxSize):
            p = contours[contourNum][j][0]
            x= p[0]
            y = p[1]
            # 这里的i对应公式中的u表示傅立叶级数的第i项,j表示第i项下的第j点
            sumx += (x * math.cos(2 * math.pi * i * j / maxSize) + y * math.sin(2 * math.pi * i * j / maxSize));
            sumy += (y * math.cos(2 * math.pi * i * j / maxSize) - x * math.sin(2 * math.pi * i * j / maxSize));
        # 这里直接把a(u)做了||a(u)||操作，方便后续归一化操作
        f.append(math.sqrt((sumx * sumx) + (sumy * sumy)))
    # 进行归一化，然后放入最终结果中
    for k in range(1,16):
        f[k] = f[k] / f[0];
        fd.append(f[k]);
    # 输出最终的手势特征
    return fd

# imageIn()
