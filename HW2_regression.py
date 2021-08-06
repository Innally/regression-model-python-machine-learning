# -*- encoding: utf-8 -*-
"""
@File    :   HW2_regression.py    
@Desciption:
@Modify Time      @Author    @Version   
------------      -------    --------   
2021/3/3 19:08    WangCC       1.0    
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split


def LeastRegression_equ(x, y):
    # 最小二乘法回归解析解实现
    xm = np.vstack((np.ones(x.shape[0]), x)).T  # 垂直拼接
    y = y.T
    theta = np.dot(np.linalg.inv(np.dot(xm.T, xm)), np.dot(xm.T, y))  # =XXXy (x*x.t )^-1*(x.t*y)
    print("最小二乘回归函数：y={:.4f}+{:.4f}*x".format(theta[0], theta[1]))  # 打印出截距
    plt.plot(x, theta[0] + theta[1] * x, label='least regression')
    return theta

def ridgeRegres(X,Y,lam=0.2):
    X=np.vstack((np.ones(X.shape[0]),X)).T
    xTx = np.dot(X.T,X)
    denom = xTx + np.eye(np.shape(xTx)[1])*lam
    theta = np.dot(np.linalg.inv(denom), np.dot(X.T,Y))
    print("岭回归的回归函数：y={:.4f}+{:4f}*x".format(theta[0],theta[1]))
    return theta


def MeanSquareEstimate(x, y, theta):
    y_pre = theta[0] + theta[1] * x
    return np.linalg.norm(y_pre - y) / x.shape[0]  # (y_pre-y)^2/n


def Generate_testset_reg(x, y):
    plt.scatter(x, y)  # 绘制原始数据散点图
    # 生成均值为0，方差为5，个数5的正态分布
    noise = np.random.normal(loc=0, scale=np.sqrt(1), size=5)
    # 生成测试数据集
    test_x = np.array(range(-2, 3, 1))
    # 最小二乘回归
    theta = LeastRegression_equ(x, y) # 获得theta[0]是截距，theta[1]是斜率
    # theta = ridgeRegres(x, y)  # 获得theta[0]是截距，theta[1]是斜率
    test_y = theta[0] + theta[1] * test_x + noise # noise是随机的，这是随机生成测试数据
    # 画测试数据集散点图
    plt.scatter(test_x, test_y,)
    plt.legend(["Regression curve","Train numbers","Test numbers"])
    # plt.show()
    # 计算平均训练误差
    loss = MeanSquareEstimate(x, y, theta) # use mean square estimation to estimate its loss
    # 计算平均测试误差
    loss_test = MeanSquareEstimate(test_x, test_y, theta)
    print("岭回归的平均训练误差为{:.4f}，平均测试误差为{:.4f}".format(loss, loss_test))
    return theta  # 返回的是theta[0]是截距，theta[1]是斜率


def Normalization_fun(x):
    # 特征零均值 就是让它分布在（-1，1）
    x = (x - np.mean(x, 0)) / (np.max(x, 0) - np.min(x, 0))
    return x


def Gradient_descent(x, y, w, learn_rate):
    '''
    :param x: X为 N*(D+1)的矩阵 x[:,0]全为1
    :param y: y为1*N的矩阵
    :param w: w为(D+1)*1的矩阵 w[0]=1
    :param learn_rate: 学习率
    :return: time_count:迭代次数, w:参数, err:MSE误差向量
    '''
    err = []
    err.append(100000)
    err.append(10000)

    time_count = 0 # 迭代次数
    while np.abs(err[-1] - err[-2]) > 0.000001:
        w_temp = np.zeros((w.shape[0], 1)) # 本次迭代的w值临时储存在这里
        for j in range(w.shape[0]):  # 对所有的参数赋值
            w_temp[j] = w[j] + learn_rate * np.dot((y - np.dot(x, w)).T, x[:, j]) / x.shape[0]
        w = w_temp  # 更新参数
        err.append(np.linalg.norm(np.dot(x, w) - y) ** 2 / x.shape[0])
        # print(err[-1])
        time_count += 1
        if np.abs(err[-1] - err[-2]) > 10 and time_count > 3:
            print('learn rate too big!')
            break
    plt.figure(2)
    plt.plot(err[2:])
    plt.ylabel('MSE')
    plt.xlabel('times')
    plt.show()
    return time_count, w, err


def Stochastic_gradient_descent(x, y, w, learn_rate):
    # X为 N*(D+1)的矩阵 x[:,0]全为1
    # y为N*1的矩阵
    # w为(D+1)*1 w[0]=1
    err = []
    err.append(100000)
    err.append(10000)
    y = y.reshape(y.shape[0], 1)
    time_count = 0
    while np.abs(err[-1] - err[-2]) > 0.00001:
        err_temp = 0
        for i in random.sample(range(x.shape[0]), x.shape[0]):  # 对排序后样本
            w_temp = np.zeros((w.shape[0], 1))
            for j in range(w.shape[0]):  # 对所有参数求梯度
                w_temp[j] = w[j] + learn_rate * (y[i] - np.dot(x[i, :], w)) * x[i, j]  # 求新的参数
            w = w_temp  # 更新参数
            err_temp += (np.dot(x[i, :], w) - y[i]) ** 2
        err.append(err_temp / x.shape[0])  # 计算MSE
        # print(err[-1])
        time_count += 1
        if np.abs(err[-1] - err[-2]) > 10 and time_count > 100000:
            print('learn rate too big!')
            break

    plt.figure(2)
    learn_rateFormat ="learn_rate="+str(learn_rate)
    plt.plot(err[2:])
    plt.ylabel('MSE')
    plt.xlabel('times')
    plt.legend([learn_rateFormat])
    plt.show()
    return time_count, w, err


def Train(x, y, w_init, learn_rate, methods):
    # x N D+1
    # y N 1
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1,test_size=0.2) # random_state 参数相当于给了一个固定的种子，使每一次分的结果都一样
    if methods == 'SGD':
        time_count, w, err = Stochastic_gradient_descent(x_train, y_train, w_init, learn_rate)
    else:
        time_count, w, err = Gradient_descent(x_train, y_train, w_init, learn_rate)
    mse_train = np.linalg.norm(np.dot(x_train, w) - y_train) ** 2 / x_train.shape[0] #训练误差
    mse_test = np.linalg.norm(y_test - np.dot(x_test, w)) ** 2 / x_test.shape[0]#预测误差
    return mse_train, mse_test, w, err


def Reg_sklearn_result(x, y):
    from sklearn import linear_model
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    # 创建模型并拟合
    model = linear_model.LinearRegression()
    model.fit(x_train, y_train)

    # 训练损失和预测损失
    resid_train = np.linalg.norm(y_train - model.predict(x_train))**2/x_train.shape[0]
    resid_test = np.linalg.norm(y_test - model.predict(x_test))**2/x_test.shape[0]
    print('sklearn train error:', resid_train, 'sklearn test error:', resid_test)
    print('w(0):',model.intercept_,'w:',model.coef_)
    return model.intercept_, model.coef_, resid_train, resid_test


if __name__ == "__main__":

    # 基本要求第一小问
    # 读取数据
    Data1 = pd.read_csv('dataset_regression.csv')
    X = np.array(Data1.iloc[:, 1])
    Y = np.array(Data1.iloc[:, 2])
    THETA_1 = Generate_testset_reg(X, Y)

    # 基本要求第二小问
    print("\nWine quality regression:")
    Data2 = pd.read_csv('winequality-white.csv')
    X2 = np.array(Data2.iloc[:, 0:-1])  # N D   0~-1是从第一列到倒数第二列，因为倒数第一列是标签
    X2 = Normalization_fun(X2)  # 这里是把x2正则化，使其分布在【-1,1】之间
    X21 = np.vstack((np.ones(X2.shape[0]), X2.T)).T  # 在X左侧添加一列全为1的列再将其转置，为什么要加一列1能（转置后是一行）
    Y2 = np.array(Data2.iloc[:, -1]) # 这里如果不reshape的话，他的维度是（X2.shape[0],）
    Y2=Y2.reshape(X2.shape[0], 1)  # 1 N
    Learn_rate = 0.1  # 梯度0.1，SGD 0.001
    # print('learn rate=', Learn_rate)
    W_init = np.random.randn(X2.shape[1] + 1, 1) # 初始化一个 W，W 的值是随机生成的，

    alpha = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    for i in alpha:
        Learn_rate = i
        print('learn rate=', Learn_rate)
        MSE_Train, MSE_Test, W, ERR = Train(X21, Y2, W_init, Learn_rate, 'SGD')
        print('train error:', MSE_Train)
        print('test error:', MSE_Test)
        print('W:', W)

    # print('train error:', MSE_Train)
    # print('test error:', MSE_Test)
    # print('W:', W)

    # 包的解
    # _, _, MSE_Train_sk, MSE_Test_sk = Reg_sklearn_result(X2, Y2)
    '''
    train 
    0.5736591700949722 
    test 
    0.5329937846330011
    w 5.87796976 0.53891102 -2.01323151  0.01072902  4.9964319  -0.22927381
   0.90448403 -0.04543689 -7.06206365  0.71422975  0.46949966  1.28538239]]
    '''

