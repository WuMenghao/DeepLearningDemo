# -*- coding: utf-8 -*-
"""
tensorflow读取机制
@author: Administrator
"""
# In[]
import tensorflow as tf

# In[]
# 需要读取图片的文件名
fileName = ['A.jpg','B.jpg','C.jpg']
# string_input_producer会产生一个文件名队列 shuffle=False时是无序的
fileName_queue = tf.train.string_input_producer(fileName,shuffle=False,num_epochs=5)
# tf.WholeFileReader获取reader
reader = tf.WholeFileReader()
# reader的使用read方法读取数据
key, value = reader.read(queue=fileName_queue)

with tf.Session() as sess:
    # 初始化tensorflow变量
    tf.local_variables_initializer().run()
    # start_queue_runners被调用之后才会开始填充队列
    tf.train.start_queue_runners(sess=sess)
    
    i = 0
    while True:
        i += 1
        # 获取图片并保持
        image_data = sess.run(value)
        with open('read/test_%d.jpg' % i,'wb') as f:
            f.write(image_data)

