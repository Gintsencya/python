#1.完成以下操作
# (1) 创建一个一维的numpy数组x保存随机产生某专业160名学生某门课期末考试的成绩（百分制）。
# (2) 分别求分数的最大值、最小值、均值、中位数、标准差、方差、第25个百分位数、第75个百分位数。
# (3) 用pandas的describe()函数验证上一步的运算是否准确，注意numpy和pandas在标准差处理上的差异。
# (4) 分别用z-score变换、最小最大、十进制尺度变换3种方法进行标准化。
# (5) 使用sklearn库的StandardScaler, MinMaxScaler实现z-score变换和归一化操作。
# (6) 将随机挑选的5处学生成绩设置为np.nan，然后用非nan的学生成绩四舍五入之后的平均分填充这5处nan。
# (7) a = np.array(np.arange(0,8).reshape(2, 4)), b = np.ones((2, 4)),分别用concatenate，hstack和vstack函数将2个二维数组在行、列上拼接为1个二维数组。能清晰解释axis轴的概念，并正确分析运行结果。
# (8) 求上一步vstack之后的数组每一列的均值，每一行的和。
# (9) 将第（7）步 vstack之后的结果 reshape为n行1列的二维数组。
# (10) 修改第（7）步的数组a的每一个元素为其平方, 修改b的每个元素使其加上5，再求a和b对应位置元素乘积。将矩阵b转置,求a和b的矩阵乘结果。
# import numpy as np
# from random import randint

# list_1 = []
# for i in range(0,160):
#     list_1.append(randint(0, 100))
# array = np.array(list_1)

# unm_max = np.amax(array)#最大值
# unm_min = np.amin(array)#最小值
# unm_median = np.median(array)#中位数
# unm_mean = np.mean(array)#平均值
# unm_std = np.std(array)#标准差
# unm_var = np.var(array)#方差
# top_25 = array[24]#第25个百分位数
# top_75 = array[74]#第75个百分位数

# import pandas as pd
# pd.Series()'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# (1) 创建一个一维的numpy数组x保存随机产生某专业160名学生某门课期末考试的成绩（百分制）。
x = np.random.randint(0,101,160)
print(x)

# (2) 分别求分数的最大值、最小值、均值、中位数、标准差、方差、第25个百分位数、第75个百分位数。
x.max()
x.min()
x.mean()
np.median(x)
x.std()
x.var()
np.percentile(x, 25)
np.percentile(x, 75)

# (3) 用pandas的describe()函数验证上一步的运算是否准确，注意numpy和pandas在标准差处理上的差异。
import pandas as pd
df = pd.Series(x)
df.describe()

# (4) 分别用z-score变换、最小最大、十进制尺度变换3种方法进行标准化。
x_norm1 = (x - np.mean(x))/np.std(x) #x_norm的均值为0，标准差为1
print(x_norm1)
x_norm2 = (x- x.min()) / (x.max() - x.min())
print(x_norm2)
x_norm2 = x / 1e2 #分⺟取10^j, j取max(x_norm) <= 1的最⼩整数
print(x_norm2)
# (5) 使用sklearn库的StandardScaler, MinMaxScaler实现z-score变换和归一化操作。
# StandardScaler和MinMaxScaler是两种不同的数据标准化方法
ss = StandardScaler()
ss.fit_transform(x.reshape(-1,1))
mms = MinMaxScaler()
mms.fit_transform(x.reshape(-1,1))

# (6) 将随机挑选的5处学生成绩设置为np.nan，然后用非nan的学生成绩四舍五入之后的平均分填充这5处nan。
score = x.astype('float')
score[np.random.randint(0,160,5)] = np.nan
mean = round(np.nanmean(score))
print('平均值：%s' %mean)
score[np.isnan(score)] = mean
print(score)
#df = pd.DataFrame(score)
#df.fillna(np.random.randint(0,160,5)) = np.nan

# (7) a = np.array(np.arange(0,8).reshape(2, 4)), b = np.ones((2, 4))
# 分别用concatenate，hstack和vstack函数将2个二维数组在行、列上拼接为1个二维数组。
# 能清晰解释axis轴的概念，并正确分析运行结果。
a = np.array(np.arange(0,8).reshape(2,4))
print(a)
b = np.ones((2,4))
print(b)
np.concatenate((a,b))
np.concatenate((a,b),axis=1)#axis=1 表示沿着第二个轴（即列轴）进行连接。
np.hstack((a,b))
np.vstack((a,b))

# (8) 求上一步vstack之后的数组每一列的均值，每一行的和。
data = np.vstack((a,b))
np.mean(data,axis=0)
np.sum(data,axis=1)

#(9) 将第（7）步 vstack之后的结果 reshape为n⾏1列的⼆维数组
data.reshape(16,1)#16行1列
data.reshape(-1,1)#n行1列

# #(10) 修改第（7）步的数组a的每⼀个元素为其平⽅, 修改b的每个元素使其加上5， 再求a
# 和b对应位置元素乘积。将矩阵b转置,求a和b的矩阵乘结果
a = np.power(a,2)#幂运算
b = 5 + b#标量与向量相加，结果是向量
a*b#对应位置元素乘积
np.multiply(a,b)#对应位置元素乘积a[0]*b[0]=c[0]
np.dot(a, b.T)#或写为np.dot(a, np.transpose(b))

# 2.有⼀组数据[-0.7, -0.5, 3.3, 0.0, -0.7] 设计⼀个softmax函数，将这组数据
# 变为概率分布，要求概率分布的和为整数1， 保留⼩数点后2位有效数字。 公式为：f(x) =e^x/sum(e^x)
x = np.array([-0.7, -0.5, 3.3, 0.0, -0.7])
def softmax(x):
    exp_x = np.exp(x)
    s = np.sum(exp_x)
    y = exp_x / s
    y_around = np.around(y, decimals=2)
    return y_around
print(softmax(x))

# Import matplotlib, numpy and math
import matplotlib.pyplot as plt
import numpy as np
import math
#sigmoid函数
x = np.linspace(-10, 10, 100)
z = 1/(1 + np.exp(-x)) 
plt.plot(x, z,'r-.',label='sigmoid')
plt.xlabel("x")
plt.ylabel("Sigmoid(X)")
plt.title('sigmoid函数')
#plt.savefig('path\simgmoid.png')#保存地址
plt.show()
#tanh函数
i = np.linspace(-np.pi, np.pi, 100)#np.pi=3.1415926
out1 = np.tanh(i)
plt.plot(i,out1,color='red',marker = "")#marker附加备注
plt.title('tanh函数')
plt.show()
#3.将sigmoid和tanh函数绘制在⼀张图上，⽤图例、线形、颜⾊等区分两条曲线。
#略，参考讲稿源码
i = np.linspace(-np.pi, np.pi, 100)
z = 1/(1 + np.exp(-x)) 
out1 = np.tanh(i)
plt.plot(i,z,'r--')
plt.plot(i,out1,'k:')
plt.legend(labels=['sigmoid函数','tanh函数'])
# plt.legend(handles=[z,out1],labels=["sin function","cos function"],loc="lower right",fontsize=6),不是很懂
plt.title('sigmoid和tanh')
plt.xlabel('xlabel')
plt.ylabel('ylabel')
# plt.savefig('path\sigmoid和tanh函数.png')
plt.show()

# 4.对鸢尾花数据集进⾏主成分降维并⽤散点图可视化。
# 略，参考讲稿源码
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

plt.rcParams['font.sans-serif'] = ['SimHei']#识别中文避免乱码
plt.rcParams['axes.unicode_minus'] = False

iris = load_iris()#鸢尾花数据集
x = iris.data
y = iris.target

data_pca = PCA(n_components=2)
reduce_x = data_pca.fit_transform(x)#降维
#(2)将类标签由int转为字符串数组target_colors
target_colors = y.astype('str')
#(3)将target_colors中的'0','1','2'转'r','g','b'
np.putmask(target_colors, target_colors == '0', 'r') 
# putmask(a,mask, values)输入数组，需要被替换的值所在的数组布尔掩码数组，指定替换条件替换的值
np.putmask(target_colors, target_colors == '1', 'g')
np.putmask(target_colors, target_colors == '2', 'b')
''' 也可⽤布尔数组筛选后进⾏替换
#target_colors[target_colors == '0'] = 'r'
#target_colors[target_colors == '1'] = 'g'
#target_colors[target_colors == '2'] = 'b'
'''
#(4)调⽤plt.scatter绘制散点图
plt.scatter(reduce_x[:,0], reduce_x[:,1], c = target_colors)
#plt.savefig('.\配图\鸢尾花PCA散点图.png')
plt.show()

# 5.TIOBE排⾏榜 https://www.tiobe.com/tiobe-index/ 是根据互联⽹上有经验的程
# 序员、课程和第三⽅⼚商的数量，并使⽤搜索引擎（如Google、Bing、Yahoo!）以及
# Wikipedia、Amazon、YouTube和Baidu（百度）统计出的计算机语⾔排名数据,其结果作为
# 2
# Wikipedia、Amazon、YouTube和Baidu（百度）统计出的计算机语⾔排名数据,其结果作为
# 当前业内程序开发语⾔的流⾏使⽤程度的有效指标。请绘制⼀张饼图展示各种语⾔的流⾏趋势
# （占⽐较少的语⾔可统⼀⽤“其它”代替）。
# #略，参考讲稿饼图绘制⽅法
import requests
import matplotlib.pyplot as plt
import re
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']#识别中文避免乱码
plt.rcParams['axes.unicode_minus'] = False

def web():#请求头
    cookies = {
        '_gid': 'GA1.2.474640489.1681110059',
        'PHPSESSID': '6nfhino97jjtf1ig5tok3o66ne',
        '_ga': 'GA1.1.1236505891.1681110058',
        '_ga_20DE9XQ5B5': 'GS1.1.1681110057.1.1.1681110763.0.0.0',
    }

    headers = {
        'authority': 'www.tiobe.com',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'accept-language': 'zh-CN,zh;q=0.9',
        'cache-control': 'max-age=0',
        # 'cookie': '_gid=GA1.2.474640489.1681110059; PHPSESSID=6nfhino97jjtf1ig5tok3o66ne; _ga=GA1.1.1236505891.1681110058; _ga_20DE9XQ5B5=GS1.1.1681110057.1.1.1681110763.0.0.0',
        'sec-ch-ua': '"Chromium";v="112", "Google Chrome";v="112", "Not:A-Brand";v="99"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'none',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
    }
    url = 'https://www.tiobe.com/tiobe-index/'
    response = requests.get(url, cookies=cookies, headers=headers)#获取网页
    content = response.text#转化为text文档类型
    return content

def line(content):
    pattern = 'class="tiobe-index container.*?<tbody>(.*?)</tbody>'
    language_pattarn = re.compile(pattern,re.S)
    result = re.findall(language_pattarn,content)
    end = re.findall('<td>(.*?)</td>', str(result))
    language = end[3:len(end):6]#筛选语言名称
    percentage = end[4:len(end):6]#筛选语言使用率
    increase = end[5:len(end):6]#筛选语言使用增长率
    interface = [language,percentage,increase]#构成二维矩阵
    return interface

def pie_fig(interface):
    numbe = []
    for i in interface[1]:#获取interface矩阵第2行#二维转化为一维
        txt = re.findall('(.*?)%',i)
        numbe.extend(txt)
    number = list(map(float,numbe))#map类型再转成列表list
    labels = interface[0]
    explode = np.linspace(0, 0.2,20)
    plt.pie(number,explode=explode,colors=None,labels=labels,autopct='%1.2f%%',shadow=0.1)
    # autopct显示百分比'%1.2f%%'为表达式，小数点后表示保留几位小数，前面的数值没影响
    plt.title('饼图')
    plt.legend(loc = (1.1,0.5))#图例loc控制图例
    plt.show()

if __name__ == '__main__':
    content = web()
    interface = line(content)
    pie_fig(interface)

#(1) 试写出运行结果
import pandas as pd
countries = ['USA','China','France','Nigeria']
my_data = [100, 200, 300, 400]
pd.Series(my_data, countries)
# pd.Series(data=None,index=None,dtype: 'Dtype | None' = None,name=None,copy: 'bool' = False,fastpath: 'bool' = False,)

# (2) 试写出运行结果
import numpy as np
np_arr = np.array(my_data)
pd.Series(np_arr)

# (3) 试写出运行结果
x = pd.Series([1, 2, 3, 4], ['a', 'b', 'c', 'd'])
y = pd.Series([2, 4, 6, 8], ['a', 'c', 'd', 'e'])
print(x + y)

# (4) 试写出运行结果
x.dtype,x.shape, x.ndim, x.index, x.values

# (5) 试写出运行结果
x[x >= 2]#按值索引

# (6)将numpy中的四行五列的[0,1)随机数数组转换为pandas中的DataFrame。
import numpy as np
import pandas as pd

target = np.random.rand(4,5)
df = pd.DataFrame(target)

# (7) 求第(6)步创建的数组的皮尔森相关系数矩阵。
t = df.corr()#皮尔逊相关系数矩阵可以用于分析这个数据集中不同变量之间的线性相关性。
print(t)

# (8) 求每一列的均值，每一行各元素的平方和。
df_mean = df.mean(axis=0)
df_apply = df.apply(lambda x:np.sum(x**2),axis=1)#对每个元素都使用了lambda表达式

# (1)创建一个字典，存储学生的id、姓名、数学成绩、英语成绩。字典的键构成cloumns，字典的值为列表。将这个列表转换为DataFrame，命名为data，表格如下所示。
#    id    name  math  english
# 0   1   Alice    90       89
# 1   2    Bob     89       94
# 2   3   Cindy    99       80
# 3   4    Eric    78       94
# 4   5   Helen    97       94
# 5   6   Grace    93       90
import numpy as np
import pandas as pd

tuple_a = {'id':[1,2,3,4,5,6],
           'name':['alice','bob','cindy','eric','helen','grace'],
           'math':[90,89,99,78,97,93],
           'english':[89,94,80,94,94,90]}

data = pd.DataFrame(tuple_a)

# (2) 选取data的name和enghlish两列；选取data的第0、2、4行
# 选取 name 和 english 两列
name_english = data[['name', 'english']]
# 选取索引为 0、2、4 的行
rows = data.iloc[[0, 2, 4]]

# (3) 增加一行数据: 7, 'Tom', 91, 89
data.loc[data.shape[0]] = [7, 'Tom', 91, 89]
print(data)
# data.shape[0]写为len(data)也可，都是获取⼆维数组的⾏数。注意本例中不能写为：
# data.iloc[data.shape[0]] = [7, 'Tom', 91, 89]。给DataFrame增加⾏时，
# pandas推荐⽤concat函数完成:
x = pd.DataFrame(np.array([7, 'Tom', 91, 89]).reshape(1,-1),columns=data.columns)
pd.concat([data, x], ignore_index=True)

#(4) 删除⼀⾏数据：将Eric这⾏数据删除
data.drop(index=data[data['name'] == 'eric'].index, inplace=True)
#drop函数既可以删除⾏也可删除列，如果不想通过axis设定0或1表示按⾏或列删除,可直接
# ⽤index或columns参数来指定删除对象，此时不⽤再设置axis参数；inplace=True表示原地修改
print(data)

data.reset_index(inplace=True, drop=True) 
#重新设置⾏索引, drop=True表示data的原inde值不作为新的⼀列插⼊到data中。
# drop的默认值为false，因为panas担⼼重置index后，原index的信息丢失，所以做了保守处理。
print(data)

#(5) 删除id这⼀列， drop和pop⽅法⼆选⼀
data.drop(labels='id', axis=1, inplace=True) #labels参数指定为'id'，aixs=1表示删除的是列
data.pop('id')# 原地修改，不设定axis和inplace参数，函数返回“被删除”的列，语法更简洁，推荐使⽤
print(data)

#(6) 增加⼀列physics，并赋初始值为0，以下两种⽅法均可
data['physics'] = 0 #⽆中⽣有法
data.insert(loc=3,column='physics',value=0) #注意insert不⽤于插⼊⾏，只⽤于插⼊列;
# 另外insert函数对data原地修改，不能设定inplace参数为True或False。

#(7) 查询Alice和Grace同学的所有信息，以下⼏种⽅法均可
data[(data['name'] == 'alice') | (data['name'] == 'grace')]
#或：
data.query('name == "alice" or name == "grace"')
#如果需要将查询结果合并：
pd.concat([data[data['name']=='alice'], data[data['name']=='grace']])

#(8) 将physics列的值设置为math和english的均值，然后增加⼀列求总分的列total,按总分、数学两项降序排列
data.loc[:,'physics'] = np.round((data['math'] + data['english'])/2).astype('int64') 
#均值⽤np.mean(data[['math','english']], axis=1)求也可。np.round求⼗进制数的四舍五⼊形式
data.loc[:,'total'] = np.sum(data.iloc[:,1:4], axis=1) #求总分
print(data)

data.sort_values(by = ['total','math'], ascending = False,inplace=True) 
#按总分、数学两项降序排列,总分⼀样再⽐较数学
print(data) 
#index仍为原先的值，如果想索引改为重新编号，可以在sort_values时，指定ignore_index=True参数，或排序号再调⽤reset_index



# 9. 安装运行MongoDB数据库，安装pymongo扩展库。
# 编写程序连接MongoDB数据库服务器；在MongoDB中创建一个myDB数据库，在该数据库下创建一个名字为myCollection的集合（集合相当于关系数据库中的表）；
# 再在该集合下创建两个文档（文档相当于关系数据库中的记录）分别存储tom和alice的个人信息。查询集合的所有文档、修改文档的内容，最后删除指定条件的文档。
# -*- coding: utf-8 -*-
'''
【基本概念】
关系型数据库： 数据库 -----> 表 -----> 记录（字段名和字段值）
MongoDB数据库： 数据库 -----> 集合 -----> ⽂档（键：值对）
【相关⽅法】
增：insert_one()、insert_many()
删：delete_one()删除第⼀个匹配到⽂档；delete_many()批量删除匹配⽂档
改：update_one(),update_many()
统计⽂档数：count_documents()
查: find_one, find()
【BSON⽂件格式】
BSON是⼀种类JSON的⼀种⼆进制形式的存储格式，BSON有三个特点：轻量性、可遍历性、⾼
效性。MongoDB使⽤了BSON这种结构来存储数据和⽹络数据交换。
'''
import pymongo
from bson import ObjectId
print(pymongo.__version__)
'''
4.2.0
由于mongodb数据库版本升级较快，旧版本的⼀些语句在新版本中已弃⽤有警告或直接产⽣异
常，做实验时请注意检查版本。本例中mongdb的版本为6.0.1
'''
# 1.连接数据库服务器,获取客户端对象
mongo_client = pymongo.MongoClient(host = 'localhost', port = 27017)
# 2.获取数据库对象
db = mongo_client.myDB
# 也可写为：db = mongo_client['myDB']
# 3.获取集合对象 ---- collection对应于关系型数据库中的table
my_collection = db.myCollection
# 也可写为：my_collection = db['myCollection']
print("——"*50)
# 4.插⼊⽂档
tom = {'name':'Tom','age':18,'gender':'男','hobbies':['吃饭','睡觉','玩游戏']}
alice = {'name':'Alice','age':19,'gender':'⼥','hobbies':['读书','跑步','学Python']}
tom_id = my_collection.insert_one(tom)
alice_id = my_collection.insert_one(alice)
print(tom_id)
print(alice_id)
print("——"*50)

print(tom_id.inserted_id)
'''623564019b3c522b9cb0a2bb'''
# 5.查询⽂档
print(my_collection.count_documents({})) # 获取⽂档个数
''' 运⾏结果：
2
'''
cursor = my_collection.find()
for item in cursor:
    print(item)
    print("——"*50)

#根据_id查询单条记录,以下两条语句均可实现，如果提示ObjectId‘ is not defined’，请检查之前是否有from bson import ObjectId
my_collection.find_one({'_id':ObjectId('623564019b3c522b9cb0a2bb')})
my_collection.find_one({'_id':tom_id.inserted_id})
''' 查询结果是⼀个dict
{'_id': ObjectId('623564019b3c522b9cb0a2bb'),
'name': 'Tom',
'age': 18,
'gender': '男',
'hobbies': ['吃饭', '睡觉', '玩游戏']}
'''
#查询年龄⼤于等于19的⽂档
for i in my_collection.find({'age':{'$gte':19}}):
    print(i)
    print("——"*50)

# 6.修改⽂档
my_collection.update_one({'name':'Tom'},{'$set':{'hobbies':['向Alice学习读书','跟Alice⼀起跑步','同Alice⼀块学习Python']}})
#$set：只修改指定键的值，其它键值对不变，否则没有修改的其它键值对也被删掉了(除了_id键值对)
#查看修改后的效果
for item in my_collection.find():
    print(item)
    print("——"*50)

# 7.删除⽂档
my_collection.delete_one({'name':'Tom'})
for item in my_collection.find():
    print(item)

#8.删除所有⽂档
my_collection.delete_many({})
''' 运⾏结果：
{'n': 1, 'ok': 1.0}
'''
#9. 删除数据库
mongo_client.db.command("dropDatabase")
''' 运⾏结果：
{'ok': 1.0}
'''
'''
MongoDB窗⼝对应的命令为：
删除集合：db.collection名.drop()
删除数据库：db.dropDatabase()
'''