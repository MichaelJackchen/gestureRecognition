import cv2
import numpy as np
import csv
import os
import imageHandle as IH

def predictImage(model, fileName):

    image = cv2.imread(fileName)
    biImage = IH.skinMask2(image)
    oneFeatures = IH.getHandFeatures(biImage)
    oneFeatures = oneFeatures[1:]
    out = np.zeros((1,14),dtype='float32')
    for i in range(0,oneFeatures.shape[0]):
        out[0][i] = oneFeatures[i]
    prediction = model.predict(out)
    return prediction[0]

def estimateModel():
    '''
    评估实验模型，使用PRF评测指标分别评估6个手势
    :return:
    '''
# 手势识别的主流程及模型评估
if __name__ == "__main__":

    trainedModel = cv2.ml.ANN_MLP_load('ann_param2')#载入训练好的模型
    # 测试数据集为后50组
    pathName = 'images'+'\\'
    # 读取训练样本
    for i in range(0,6):
        files = os.listdir(pathName+str(i))  # 得到文件夹下的所有文件名称
        for j in range(150,200):
            filename = pathName+str(i)+'\\'+files[j]
            kk = predictImage(trainedModel,filename)
            print('应该是'+str(i)+'预测的：'+str(kk))

