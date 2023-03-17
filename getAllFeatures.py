import cv2 as cv
import os
import imageHandle as IH
import csv

if __name__ == "__main__":
    '''
    提取所有训练样本的特征，按照特征值与标签矩阵形式存储
    '''
    file_xml = 'imageFeatures.xml'
    pathName = 'images'+'\\'
    # 读取训练样本
    for i in range(0,6):
        files = os.listdir(pathName+str(i))  # 得到文件夹下的所有文件名称
        f = open('featuresData'+'/'+str(i)+'.csv', 'w', encoding='utf-8',newline='')
        csv_writer = csv.writer(f)  #构建csv写入对象
        for j in range(0,100):
            image = cv.imread(pathName+str(i)+'\\'+files[j])
            biImage = IH.skinMask2(image)
            oneFeatures = IH.getHandFeatures(biImage)
            csv_writer.writerow(oneFeatures.tolist()[1:])
            print('正在处理手势'+str(i)+'_'+files[j])
        f.close()
