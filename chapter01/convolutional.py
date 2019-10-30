# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 13:42:26 2019
手写数字识别:
    两层卷积神经网络
@author: Administrator
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# wieght_variable() bias_variable()两个函数可以分别用来创建卷积核(kernel)与偏置(bias)
# 权重变量: 返回一个给定形状的变量，并自动以截断正态分布初始化
def wieght_variable(shape):
    inital = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(inital)

# 偏差变量: 返回一个给定形状的变量，初始化所有值是0.1
def bias_variable(shape):
    inital = tf.constant(0.1,shape=shape)
    return tf.Variable(inital)

# 真正进行卷积运算，卷积运算后用ReLU作为激活函数(激励函数)
# 激活函数可以引入非线性因素，解决线性模型所不能解决的问题
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 池化
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], 
                          strides=[1, 2, 2, 1], padding='SAME')


if __name__ == "__main__":
    #(1)读取数据
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # x为训练图像占位符、y_为训练图像标签的占位符
    x = tf.placeholder(tf.float32,[None,784])
    y_ = tf.placeholder(tf.float32,[None,10])
    # 使用卷积神经网络需要将单张图片还原为28x28的矩阵图片[-1,28,28,1]
    # -1表示第一维的大小是根据x自动确定的
    x_image = tf.reshape(x, [-1,28,28,1])
    
    #(2)第一层卷积层
    W_conv1 = wieght_variable([5, 5, 1, 32])#W
    b_conv1 = bias_variable([32])#b
    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1))#卷积计算
    h_pool1 = max_pool_2x2(h_conv1)#池化
    
    #(2)第二层卷积层
    W_conv2 = wieght_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    #(3)全连接层，输出为1024维向量
    #在全连接层中加入了Dropout，它是防止神经网络过拟合的一种手段
    #在每一步训练时，以一定概率'去掉'网络中的某些连接，但这种去除
    #不是永久性的，只是在当前步骤中去除，并且每一步去除的连接都是随机选择的
    W_fc1 = wieght_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1= tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # 使用Dropout, keep_prob是一个占位符, 训练时为0.5，
    #去除的概率为0.5， 测试时为1全部保留
    keep_prod = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prod)
    
    #(4)输出层全连接, 把1024维的向量转换成10维，对应10个类别
    # y_conv相当于Softmax模型中的Logit, 使用Softmax函数将其转换为10个类别的概率
    # 最后定义交叉熵损失
    W_fc2 = wieght_variable([1024,10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop,W_fc2) + b_fc2
    
    #(5)计算交叉熵损失，不采用先计算Softmax再计算交叉熵的方法
    #直接采用 tf.nn.sigmoid_cross_entropy_with_logits 直接计算
    cross_entropy = \
        tf.nn.sigmoid_cross_entropy_with_logits(logits=y_conv,labels=y_)
    
    #(6)使用梯度下降法正对模型参数(W和b)进行优化
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    #(7)计算准确率
    correct_predoction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_predoction,tf.float32))
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            barch = mnist.train.next_batch(50)
            # 每100卜报告测试一次准确率
            if i%100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                        x: barch[0],y_: barch[1],keep_prod:1.0})
                print("step %d, training accuracy %g" % (i,train_accuracy))
            # 进行训练
            train_step.run(feed_dict={x: barch[0],y_: barch[1],keep_prod:0.5})
        #(8)测试
        print("test accuracy %g" % accuracy.eval(feed_dict={
                x:mnist.test.images, y_:mnist.test.labels, keep_prod: 1.0}))
