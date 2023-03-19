import cv2
import numpy as np
import csv
import imageHandle as IH

def trainModel():
    # 读取特征数据
    data = np.empty([0,14],dtype='float32')
    for i in range(0,6):
        f = open('featuresData' + '/' + str(i) + '.csv', 'r', encoding='utf-8')
        csv_reader = csv.reader(f)  # 使用csv.reader读取csvfile中的文件
        for row in csv_reader:  # 将csv 文件中的数据保存到data中
            row_values = np.expand_dims(np.array(row), axis=0)
            data = np.append(data, row_values, axis=0)
        f.close()
    #
    data = np.array(data,dtype='float32')
    imageNum = data.shape[0]
    # 特征标签：告诉神经网络对应手势的特征数字 600x6：对应相同数字时列为1，其余为0，比如第120张图片为手势1的特征，则对应列向量1的标签为1，其余为0
    trainClasses = np.zeros((imageNum,6,1),dtype=np.float32)
    for i in range(0,trainClasses.shape[0]):
        trainClasses[i][int(i/150)] = 1

    model = cv2.ml.ANN_MLP_create()#建立模型
    # 14：输入层神经元节点   24：隐藏层神经元节点     6:输出层节点
    model.setLayerSizes(np.int32([14,24,6]))
    # 0.1:权梯度项强度,动量项强度
    model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP,0.1,0.1)#设置训练方式为反向传播
    model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)#设置激活函数为SIGMOID
    # 5000：迭代次数，0，0001：误差最小值
    model.setTermCriteria((cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 5000, 0.01 ))#设置终止条件
    # 训练模型
    model.train(data, cv2.ml.ROW_SAMPLE, trainClasses)
    # 保存模型
    model.save('ann_param2')
    print("模型训练完成！")
    # model.setTermCriteria(( cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 500, 0.0001 ))#设置终止条件
    # model.setBackpropWeightScale(0.001)  #设置反向传播中的一些参数
    # model.setBackpropMomentumScale(0.0) #设置反向传播中的一些参数

#构建模型网络，一会进行调用
# class NeuralNetwork(object):
#     def __init__(self, file_path):
#         self.model = cv2.ml.ANN_MLP_load(file_path)#载入训练好的模型
#     #预测拍的照片给出标签值
#     def predict(self, samples):
#         ret, resp = self.model.predict(samples)
#         return resp.argmax(-1)

# trainModel()