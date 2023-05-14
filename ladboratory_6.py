# （1）鸢尾花(iris)数据集包含150个样本数据，分为3类（山鸢尾，变色鸢尾，维吉尼亚鸢尾），每类50个样本。每个样本包含4个属性（特征）。
#   从sklearn库读取该数据集，用train_test_split函数拆分为训练集和测试集。
#   搭建全连接网络模型，实现模型的训练，可视化acc和loss曲线。使用该模型对测试集中的鸢尾花样本进行类别预测，评估分类效果。
import tensorflow as tf
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split#划分测试集和训练集。tensorflow acc loss  -*- coding: UTF-8 -*-

#1.划分数据建立测试集和训练集
X = load_iris().data    #导入数据
y = load_iris().target   #导入标签
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)#random_state种子

#2.导入训练模型
model = tf.keras.models.Sequential()   #Sequential用于顺序、无分枝、无跳连的网络结构。(输入层input layer)
model.add(tf.keras.layers.Dense(10, activation='relu'))     #Dense表示全连接层，共10个神经元，激活函数为relu(中间层hidden layer)
model.add(tf.keras.layers.Dense(3, activation='softmax'))   #神经元数为3激活函数为softmax，变为一组概率和为1的概率分布。(输出层output layer)
#扩展:用函数法搭建模型
# input = tf.keras.layers.Input(shape=(4,))
# x = tf.keras.layers.Dense(10)(input)
# x = tf.keras.layers.Activation('relu')(x)
# x = tf.keras.layers.Dense(3)(x)
# output = tf.keras.layers.Activation('sigmoid')(x)
# model = tf.keras.models.Model(inputs=input, outputs=output)

#3.编译模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])

# optimizer优化器,adam是比较常用的方法,不同学习方法只是一阶动量和二阶动量不同,请参考B站中推荐视频的讲解。
# loss损失函数,深度学习常用crossentropy即交叉嫡损失函数: -sum(y_true*ln(y_pred))。多分类用categorical_crossentropy,二分类用binary_crossentropy; sparse的意思是:原始数据集中的类标签y_true是未经one-hot编码的整数,比如0、1、2,如果类标签为one-hot编码则不加sparse
# metrics 准确率衡量指标,根据loss来选择对应的指标。也可以直接设定为[ 'accuracy'],模型会自动选择!

#4.训练模型
history = model.fit(X_train, y_train, batch_size=16, epochs=500)
#epochs=500表示迭代500次，即将训练集112个样本要反复送入500次到网络中进行训练; 
# batch_size=16表示每批数据有16个样本，这样训练集112个样本其实可视为是分成112/16=7批送入网络的。
# fit函数batch_size参数的默认值为32，即每批32样本前向送入网络后再统一算这一批数据的损失函数，然后反向更新模型参数。
# 显然，此时112/32=3.5即训练集可视为分4批送入模型。显然batch_size设定为16时要比32时训练时间更长;运行结果保存于history变量便于将来分析和绘图;
# 有时为了对比也可设定“验证集”，比如设定validation_data参数:history = model.fit(X_train， y_train， batch_size=16,epochs=500,
# validation_data=(X_test，y_test))这样将来可以训练集和验证集的训练效果和预测结果绘制在1张图中。


#5.结果评估
# 1训练过程可视化
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot()
# plt.savefig('\配图\神经网络训练过程') #保存图片
plt.show()

# 2整体准确率
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2) 
#evaLuate时传入的是测试集的数据，test_loss,test_acc均为最后结果两个标量
# verbose =0 不在标准输出流输出日志信息，即在控制台没有任何输出verbose = 1输出进度条（默认值）verbose = 2不输出进度条，为每个epoch输出一行记录
print(' %.2f%%' %(test_acc*100))

# 3预测样本
predictions = model.predict(X_test, verbose=2) #测试集38个样本，输出层3个神经元，所以predictions是38*3的二维ndarray
print(predictions[0])
print(' %d %.2f%%' %(tf.argmax(predictions[0]), max(predictions[0])*100))
print(predictions)

# 4输出混淆矩阵、分类报告
from sklearn.metrics import confusion_matrix, classification_report
y_true = y_test
y_pred = np.argmax(predictions, axis=1)
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
#查看模型结构
print(model.summary())


# （2）【重点】fashion-mnist数据集是一个包含60000张训练图像和和应标签、10000张测试图像和对应标签的数据集，用于建立学习模型来对10类不同的服饰进行预测。
# 用tensorflow搭建网络学习模型，分别用普通的全连接网络和卷积神经网络完成实验，对比实验结果。
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import tensorflow as tf
from tensorflow import keras      # 载入TensorFlow 和 tf.keras 
from sklearn.metrics import confusion_matrix, classification_report#混淆矩阵、分类报告
import os   # os模块提供的就是各种 Python 程序与操作系统进行交互的接口

#1 读取数据，理解图像
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data() #从网站下载数据
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') 
input_shape = (28, 28, 1)
#2标准化
x_train = x_train / 255.0   #像素的值除以255
x_test = x_test / 255.0
#3、搭建模型
#(1)第1个卷积层
model = keras.models.Sequential() #选择序贯模型
model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), input_shape=input_shape))
model.add(keras.layers.BatchNormalization()) 
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
#(2)第2个卷积层
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu')) 
model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
model.add(keras.layers.Dropout(0.2)) 
#(3)全连接层
model.add(keras.layers.Flatten())#Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。28*28变为1*784 Flatten不影响batch的大小。
model.add(keras.layers.Dense(128, activation = 'relu'))#添加全连接层，输出空间维度（节点）为128，激活函数为relu，作用是分类
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(64, activation = 'relu')) 
model.add(keras.layers.Dropout(0.2)) #舍弃
model.add(keras.layers.Dense(10, activation = 'softmax')) #添加全连接层，输出空间维度为10，激活函数为softmax

#4编译模型
#Adam优化器，adam有默认的学习率，所以不用写lr大小，作用：自适应动态调整学习率(lr),#loss损失函数
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics =['sparse_categorical_accuracy'])
model.summary()

#5、训练
#(1)断点续训
checkpoint_save_path = 'checkpoint_CNN/mnist.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
 print('----------load the model----------')
 model.load_weights(checkpoint_save_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path, save_weights_only=True,save_best_only=True)

#(2)训练
history = model.fit(x_train, y_train, epochs = 30, validation_data = (x_test, y_test),callbacks=[cp_callback]) 

#(3)绘制acc和lLoss曲线
def plot_learning_curves(history):
 pd.DataFrame(history.history).plot(figsize=(8,5))
 plt.grid(True)
 plt.ylim(0, 1)
 plt.show()
plot_learning_curves(history)

# (4)保存模型
model.save( 'fashion_model.h5 ')

#6、评估和预测
# 1 评测evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)#model.evaluate函数预测给定输入的输出，verbose=2 为每个epoch输出一行记录
print('\nTest accuracy:', test_acc)

# 2 预测predict 
predictions = model.predict(x_test)
print(predictions[0])
print(' %d , %.2f%%' %(np.argmax(predictions[0]), max((predictions[0]))*100))
# 预测predict
predictions = model.predict(test_images)
predictions[0]
np.argmax(predictions[0])#返回一个numpy数组中最大值的索引值。当一组中同时出现几个最大值时，返回第一个最大值的索引值。

#7、查看模型分类效果
# (1)绘制混淆矩阵
y_true = y_test
y_pred = np.argmax(predictions, axis=1)
pd.crosstab(y_true, y_pred)
confusion_matrix(y_true, y_pred)
# (2)打印分类报告
print(classification_report(y_true, y_pred))