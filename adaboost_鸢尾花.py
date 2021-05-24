from numpy import *
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def create_data():
    iris = load_iris();
    df = pd.DataFrame(iris.data, columns=iris.feature_names);
    df['label'] = iris.target;
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label'];
    data = np.array(df.iloc[:100, [0, 1, -1]]);
    for i in range(len(data)):
        if data[i,-1] == 0:
            data[i,-1] = -1
    # print(data)
    return data[:,:2], data[:,-1]

# 导入数据
data, lab = create_data()

#初始化列表，用来存放单层决策树的信息
weakClassArr = []

#获取数据集行数
m = shape(data)[0]
print(m)

#初始化向量D每个值均为1/m，D包含每个数据点的权重
D = mat(ones((m,1))/m)
#print(D)

#初始化列向量，记录每个数据点的类别估计累计值
est_val = mat(zeros((m,1)))
#print(est_val)

# 设置迭代次数
gen_max = 40;

# 开始迭代
for i in range(gen_max):

    # 开始构建单层决策树
    #print(D)

    # 初始化数据集和数据标签
    data_matrix = mat(data);
    label_mat = mat(lab).T

    # 获取行列值
    m, n = shape(data_matrix)
    #print(m, n)    # 100, 2

    # 初始化步数，用于在特征的所有可能值上进行遍历
    num_step = 10.0

    # 初始化字典，用于存储给定权重向量D时所得到的最佳单层决策树的相关信息
    best_stump = {}

    # 初始化类别估计值
    best_clas_est = mat(zeros((m, 1)))
    #print('the best_clas_est is: ', best_clas_est)

    # 将最小错误率设无穷大，之后用于寻找可能的最小错误率
    min_err = inf

    # 遍历数据集中每一个特征
    for i in range(n):

        # 获取数据集的最大最小值
        min_value = data_matrix[:, i].min()
        max_value = data_matrix[:, i].max()
        #print(min_value, max_value)

        # 根据步数求得步长
        step_size = (max_value - min_value) / num_step
        # print(step_size)

        # 遍历每个步长
        for j in range (-1, int(num_step)+ 1):  # -1 ~ 10

            # 遍历每个不等号: less than和 great than
            for inequal in ['lt', 'gt']:
                #print(inequal)

                #设定阈值,注意数据类型
                threshod = (min_value + float(j) * step_size)
                # print(threshod)

                # 新建一个数组用于存放分类结果，初始化都为1
                predict_val = ones((shape(data_matrix)[0], 1))

                # 根据阈值进行分类，并将分类结果存储到retArray
                if inequal == 'lt':
                    predict_val[data_matrix[:, i] <= threshod] = -1.0
                else:
                    predict_val[data_matrix[:, i] > threshod] = - 1.0
            # print(predict_Val)

                # 初始化错误计数向量
                predict_err = mat(ones((m, 1)))
                m, n = predict_err.shape

                # 如果预测结果和标签相同，则相应位置0
                predict_err[predict_val == label_mat] = 0

                # 计算权值误差，这就是AdaBoost和分类器交互的地方
                weight_error = D.T * predict_err
                #print("D: ", D.T)

                # 打印输出所有的值
                #print("分类: dim %d, 阈值： %.2f, ""不等号: %s, 权值误差：%.3f"
                #      % (i, threshod, inequal, weight_error))

                # 如果错误率低于minError，则将当前单层决策树设为最佳单层决策树，更新各项值
                if weight_error < min_err:
                    min_err = weight_error
                    best_clas_est = predict_val.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = threshod
                    best_stump['ineq'] = inequal
                # 最佳单层决策树构建完成

    # 根据公式计算alpha的值，max(error, 1e-16)用来确保在没有错误时不会发生除零溢出
    alpha = float(0.5 * log((1.0 - min_err) / max(min_err, 1e-16)))
    # print(alpha)

    # 保存alpha的值
    best_stump['alpha'] = alpha

    # 填入数据到列表:用来存放单层决策树的信息
    weakClassArr.append(best_stump)
    # print(weakClassArr)

    # 为下一次迭代更新D
    expon = multiply(-1 * alpha * mat(lab).T, best_clas_est)
    D = multiply(D, exp(expon))
    D = D / D.sum()
    #  print(D.T)

    # 累加类别估计值
    est_val += alpha * best_clas_est
    # print(est_val)

    # 计算错误率，aggClassEst本身是浮点数，需要通过sign来得到二分类结果
    aggErrors = multiply(sign(est_val) != mat(lab).T, ones((m, 1)))
    errorRate = aggErrors.sum() / m
    print("total error: ", errorRate)
    # 如果总错误率为0则跳出循环
    if errorRate == 0.0:
        break


# 进行分类
def adaClassify(datToClass, weakClassArr):
    #初始化数据集
    dataMatrix = mat(datToClass)
    #获得待分类样例个数
    m = shape(dataMatrix)[0]
    #构建一个初始化为0的列向量，记录每个数据点的类别估计累计值
    aggClassEst = mat(zeros((m,1)))
    #遍历每个弱分类器
    for i in range(len(weakClassArr)):
        #基于stumpClassify得到类别估计值
        # 新建一个数组用于存放分类结果，初始化都为1
        retArray = ones((shape(dataMatrix)[0], 1))
        # lt：小于，gt；大于；根据阈值进行分类，并将分类结果存储到retArray

        if weakClassArr[i]['ineq'] == 'lt':
            retArray[data_matrix[:, weakClassArr[i]['dim']] <= weakClassArr[i]['thresh']] = -1.0
        else:
            retArray[data_matrix[:, weakClassArr[i]['dim']] > weakClassArr[i]['thresh']] = -1.0

        aggClassEst += weakClassArr[i]['alpha'] * retArray

    return sign(aggClassEst)

# 初始化数据
data, lab = create_data()

# 打印输出，一半的值为1，一半的值为-1
print(adaClassify(data, weakClassArr))






