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

#1.自带模型评估
print(knn.score(X_test, y_test))
#2:使用skLearn.metrics里面的cLassification_report模块对预测结果做更加详细的分析。
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict, target_names=target_names))#返回一个评估表
#3:使用混淆矩阵查看分类效果。
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
print(cm)#[[13  0  0]
#          [ 0 15  1]
#          [ 0  0  9]]

# ===============================================================================================================================
# 2. 读取泰坦尼克号titanic.txt中的数据为DataFrame，查看数据的shape大小、用info函数和describe函数查看统计描述；
# 根据需要绘制折线图、饼图、直方图、盒图、条形图各1个（matplotlib绘图的代码熟练掌握）；
# 将age字段缺失值的乘客年龄填充为非缺失用户年龄的均值；将数据分割为训练集和测试集；
# 用KNN和其它分类算法进行建模分类；对比不同算法的分类效果，并解释运行结果。
# coding: utf-8

#导入openpyxl用于excel操作
'''
from openpyxl import Workbook
import pandas as pd

#新建保存结果的excel,sheet
wb = Workbook(r'result.xlsx')
ws = wb.create_sheet('Sheet1')

#打开txt文件,把逗号替换成统一的\t
titanic = pd.read_csv('泰坦尼克号数据集.txt')

#保存excel文件
wb.save('result.xlsx')
'''

import pandas as pd
titanic = pd.read_csv('泰坦尼克号数据集.txt')
# shape , (1313, 11)
titanic.shape
# 返回前五行
titanic.head()
# 使用info()查看各列的名称、非NaN数量、数据类型
titanic.info()
#查看缺失值情况另外一种方法
titanic.isnull().sum().sort_values()
# 查看列标签
titanic.columns
# 查看行索引
titanic.index

#数据可视化
#导入matplotlib库，设置plt.rcParams以正常显示汉字和负号
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#1直方图:年龄和生还的关系
titanic['age'].plot(kind='hist', bins=14)
plt.savefig(r'..\配图\2-1直方图')
plt.show()

#eaborn库，调用distplot函数绘制，绘制时核密度估计参数kde默认为True即同时绘制kde曲线
import seaborn as sns
sns.distplot(titanic['age'], bins=14)
plt.show()

#2饼图:绘制生还和遇难比例的
x = titanic['survived'].value_counts()
plt.pie(x,labels =[ '遇难','生还'], autopct = '%1.1f%%', startangle = 90,explode =[0,0.1])
plt.axis('equal')
plt.legend()    #配置图例
plt.savefig(r'.\配图\2-2饼图')  #保存图片
plt.show()

#3盒图:绘制生还和遇难年龄分布的盒图
#注意在满足条件的情况下取某列时的两种方法
x_0 = titanic[titanic['survived'] == 0]['age'].dropna()# 遇难人员的所有信息,取age这一列的值.dropna()去掉所有NaN
x_1 = titanic.loc[titanic['survived'] == 1,'age'].dropna()  #生还者

plt.boxplot([x_0,x_1],labels=['遇难','生还'])
plt.savefig(r'.\配图\2-3盒图')
plt.show()

# 直方图:所有生还者中年龄分布情况
x = titanic[titanic['survived'] == 1]['age'].dropna()
plt.hist(x,edgecolor = 'black',facecolor = '#4OEODO', bins = 30)
plt.savefig(r'.\配图\2-4直方图')
plt.show()

# 直方图:15岁以下生还人员年龄分布
x = titanic[(titanic['survived'] == 1) & (titanic['age']<= 15)]['age'].dropna()
# pandas用布尔矩阵进行筛选数据时，与Python语言不同没有and or not关键词，而是用&|～逻辑
plt.hist(x, edgecolor = 'black', facecolor = '#4OEODO')
plt.savefig(r'.\配图\2-5直方图')
plt.show()

#6条形图:绘制对比15岁以下人员遇难、生还情况的条形图
xo= titanic.query('survived == 0 and age <= 15')['age']
x1 = titanic.query('survived == 1 and age <= 15')['age']
plt.bar(x = 0,height = len(xo), color='red')
plt.bar(x = 1,height = len(x1), color='green')
plt.xticks([0,1],['遇难','生还'])#分别制定刻度和刻度标签
plt.savefig(r'.\配图\2-6条形图')
plt.show()

#2:性别与生还的关系
titanic.groupby(['sex', 'survived'])['survived'].count()
#遇难者中性别统计
x0 = titanic[titanic['survived'] == 0]['sex'].value_counts()#.value_counts()统计合并相同数据
#饼图生还者、遇难者性别统计
x1 = titanic[titanic['survived'] == 1]['sex'].value_counts()
plt.subplot(121)
plt.pie(x1, labels = x1.index,autopct = '%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('生还者中性别比例')
plt.subplot(122)
plt.pie(x0,labels = x0.index,autopct = '%1.1f%%', startangle=90)
plt.title('遇难者中性别比例')
plt.axis ('equal')
plt.savefig(r'.\配图\2-7饼图')
plt.show()

#3饼图:船舱等级和生还关系
x = pd.crosstab(titanic.pclass,titanic.survived,margins = True)
#创建交叉表，margins = True表示显示汇总信息
plt.subplot(131)
plt.pie(x.iloc[0, :2],labels =['遇难','生还'], autopct = '%1.1f%%')
plt.axis('equal')
plt.title('一等舱生还情况')
plt.subplot(132)
plt.pie(x.iloc[1, :2],labels =['遇难','生还'], autopct = '%1.1f%%')
plt.axis ('equal')
plt.title('二等舱生还情况')
plt.subplot(133)
plt.pie(x.iloc[2, :2],labels =['遇难','生还'], autopct = '%1.1f%%')
plt.axis('equal')
plt.title('三等舱生还情况')
plt.savefig(r'.\配图\2-8饼图')
plt.show()


#第2步，数据预处理，包括特征选择，填充nan值，数据特征化等
X = titanic[['pclass', 'age', 'sex' ]]#指定特征列
y = titanic['survived']#指定类别所在的列
X.info()
#借由info()的输出，拟设计如下几个数据预处理任务:
# 1.age这个数据列，只有633个non-null值，需要对null值进行补完。
# 2.sex 与pclass两个数据列的值都是非数值型的，需要转化为数值特征，用整数0/1，1/2/3代替。#首先补充age里的数据，使用平均数mean()或者中位数median()都是对模型偏离造成最小影响的策略。
X = X.copy() #复制一份数据，否则下一行代码可能会警告"A value is trying to be set on acopy of a slice from a DataFrame"
X['age'].fillna(X['age'].mean(), inplace = True)#inplace = True表示原地修改
#对补完的数据重新探查。
X.info()
#由此得知，age特征得到了补完。
#对客舱等级、性别分别进行特征化
X.loc[X.sex == 'male','sex'] = 0
X.loc[X.sex == 'female','sex'] = 1
X.loc[X.pclass == '1st','pclass'] = 1
X.loc[X.pclass == '2nd','pclass'] = 2
X.loc[X.pclass == '3rd','pclass'] = 3
X.columns
#将数据分割为训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state = 33)
#导入warnings库，过滤可能的警告
import warnings
warnings.filterwarnings("ignore")
#第3步，对模型进行训练，并预测，基本上是:导库-初始化-fit-predict四步
# #KNN法
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
#以下是其它算法的实现
#决策树法
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy' )
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
#随机森林法
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
#线性分类器--逻辑回归分类
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
#分类回归树CART
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
#支持向量机SVM
from sklearn.svm import LinearSVC
model = LinearSVC()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
#朴素贝叶斯
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

# 三种评估:自带评估，classification_report模块评估，混淆矩阵
print(model.score(X_test, y_test))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict, target_names = ['died', 'survived']))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_predict))