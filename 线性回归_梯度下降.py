import numpy as np
import matplotlib.pyplot as plt
import mglearn  #线性回归求斜率w和截距b，数据集使用mglearn库自带的wave数据集，pip install mglearn安装该扩展库。
from sklearn.linear_model import LinearRegression #导入线性回归模型
from sklearn.model_selection import train_test_split

X, y = mglearn.datasets.make_wave(n_samples=40)#导入wave数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)#划分训练集和测试集
