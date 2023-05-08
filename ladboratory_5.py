# 1. 用pandas从sklearn库中读取鸢尾花数据集；将150条记录分割为训练集和测试集两部分；
# 对训练和测试的特征数据进行标准化；用KNN算法进行建模分类；对分类结果分别用model.score函数
# 、classification_report、confusion_matrix三种方式进行分类效果的评估。
# （以上代码需要能背着敲出来）
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
data = pd.DataFrame(iris.data)#取出iris的data
target = pd.DataFrame(iris.target)#取出iris的target

df = pd.concat((data, target), axis=1) #拼接两个数据
df.columns = ['sl', 'sw', 'pl', 'pw', 'target']#改名
target_names = iris.target_names
df.corr()#其相关系数

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25,random_state = 0)
# 划分训练集和测试集。test_size=0.25其中25%是测试集 75%训练集。random_state = 0是随机种子
# X_train, X_test, y_train, y_test都是DataFrame秩为2

#提示:如果将train_test_split(data，target,test_size=0.25,random_state = 0)里的
# data，target改为iris.data和lris.target，则X_train,，X_test, y_tran,Yy_lesdn大生和ndarray，
# 且y_train和y_test的shape分别为:(112,)和(38,)，即秩为1,请注意区分ndarray和DataFrame的区别。

ss = StandardScaler()#数据归一化
X_train = ss.fit_transform(X_train) # X_train
X_test = ss.transform(X_test)
#提示1:ss.transform用上一步算得的x_train的均值和标准差，来对x_test数据进行标准化。思考为什么这里不能用ss.fit_transform
#提示2:X_train和x_test原先为DataFrame经过标准化之后，被转为秩为2的ndarray

knn = KNeighborsClassifier(n_neighbors=5)#导入knn模型，找出和当前这一数据相近的五个数据，对这一数据分类
knn.fit(X_train, y_train) #导入训练集，进行训练
y_predict = knn.predict(X_test)#导入测试集进行分析
print(y_predict)#测试结果

#1.自带模型评估，准确率0.9736842105263158
print(knn.score(X_test, y_test))
# 2:使用skLearn.metrics里面的cLassification_report模块对预测结果做更加详细的分析。
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict, target_names=target_names))#返回一个评估表
#3:使用混淆矩阵查看分类效果。
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
print(cm)#[[13  0  0]
#          [ 0 15  1]
#          [ 0  0  9]]


# 2. 读取泰坦尼克号titanic.txt中的数据为DataFrame，查看数据的shape大小、用info函数和describe函数查看统计描述；
# 根据需要绘制折线图、饼图、直方图、盒图、条形图各1个（matplotlib绘图的代码熟练掌握）；
# 将age字段缺失值的乘客年龄填充为非缺失用户年龄的均值；将数据分割为训练集和测试集；
# 用KNN和其它分类算法进行建模分类；对比不同算法的分类效果，并解释运行结果。

import pandas as pd
url = 'https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv'
document = pd.read_csv(url)
titanic = document.txt

# 3. 理解损失函数、梯度下降法、学习率、L1和L2正则化的基本概念。
# 安装mglearn库，使用LinearRegression、SGDRegressor、Ridge、Lasso 四种回归方法预测扩展波士顿房价，对比试验结果，解释运行结果。
from sklearn.datasets import load_boston    #load_boston 已经在 scikit-learn 的1.2版本中被移除了
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = load_boston()

# 4. 选做信用卡欺诈检测案例的实验，理解上采样、下采用的概念。