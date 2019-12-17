# 第6章 人脸检测和人脸识别

人脸检测(`Face Detection`)和人脸识别技术是深度学习的重要应用之一。本章首先
会介绍 MTCNN 算法的原理，它是基于卷积神经网络的一种高精度的实时人脸检测
和对齐技术。接着，还会介绍如何利用深度卷积网络提取人脸特征，以及如何利用
提取的特征进行人脸识别。最后会介绍如何在TensorFlow中实践上述的算法。

## 6.1 MTCNN 的原理

搭建人脸识别系统的第一步是人脸检测，也就是在图片中找到人脸的位置。在这个过
程中，系统的输入是一张可能含有人脸的图片，输出是人脸位置的矩形框。一般来说，
人脸检测应该可以正确检测出图片中存在的所有人脸，不能遗漏，也不能错检。

获得包含人脸的矩形框后，第二部要做的是人脸对齐(`Face Alignment`)。原始图片中
人脸的姿态、位置可能有较大的区别，为了之后统一处理，要把人脸"摆正"。为此，
需要检测人脸中的关键点(`Landmark`),如眼睛的位置、鼻子的位置、嘴巴的位置、脸的
位置、脸的轮廓等。根据这些关键点可以使用仿射变换将人脸统一校准，以尽量消除
姿势不同带来的误差。

这里介绍一种基于深度卷积神经网络的人脸检测和人脸对齐方法——`MTCNN`。MT是英
文单词`Multi-task`的简写，意即这种方法可以同时完成人脸检测和人脸对齐两项任务
。 相比于传统方法，MTCNN的性能更好，可以更精确地走位人脸；此外，MTCNN也可
以做到实时的检测。

MTCNN由三个神经网络组成，分别是P-Net、 R-Net、 0-Net。在使用这些网络之前，
首先要将原始图片缩放到不同尺度， 形成一个“图像金字塔”。接着会对每个尺度
的图片通过神经网络计算一遍。这样做的原因在于：原始图片中的人脸存在不同的尺
度，如奇的人脸比较大，有的人脸比较小。对于比较小的人脸，可以在放大后的图片
上检测；对于比较大的人脸，可以在缩小后的国片上检测。这样，就可以在统一的
尺度下检测人脸了。

**6.1.1 P-Net**

现在再来讨论第一个网络`P-Net`的结构,`P-Net`的输入是一个宽和高皆为`12像素`，
同时是3通道的RGB图像，该网络要判断这个`12x12`的图像中是否含有人脸，并且给出
人脸框相关键点的位置。因此，对应的输出由三部分组成：

> * 第一个部分要判断该图像是否是人脸（`face classification` ) ,输出向量的形
    状为`lxlx2`，也就是两个值，分别为该图像是人脸的概率，以及该图像不是人脸
    的概率。这两个值加起来应该严格等于`l`。之所以使用两个值来表示，是为了方
    便定义交叉烟损失。
> * 第二个部分给出框的精确位置（`boundingbox regression`),一般称之
   为框回归。P-Net输入的`12×12`的图像块可能并不是完美的人脸框的位置，如
   高的时候人脸并不正好为方形，有的时候`12×l2`的图像块可能偏左或偏右，因
   此需要输出当前框位置相对于完美的人脸框位置 的偏移。 这个偏移由四个变量组
   成。一般地， 对于圄像中的框，可以用四个数来表示E自由位置：框左上角的横坐
   标、框左上角的纵坐标、框的宽度、 框的高度。 因此，框回归输出的值是：框左
   上角的横坐标的中目对偏移、框左上角的纵坐标的相对偏移、 框的宽度的误差、
   框的高度的误差。 输出向量的形状就是圄中的`1×l×4`
> * 第三个部分给出人脸的5个关键点的位置(`Facial landmark localization`)。`5`
    个关键点分别为：左眼的位置、右眼的位置、鼻子的位置、左嘴角的位置、右嘴
    角的位置。每个关键点又需要横坐标和纵坐标两维来表示，因此输出一共是10维
    （即`l×l×10`） 。
 
图中框的大小各高不同， 除了框回归的影响外， 主要是因为将图片金字塔中的各个
尺度都使用P-Net计算了一遍， 因此形成了大小不同的人脸框。P-Net的结果还是比较
粗糙的， 所以接下来又使用R-Net进一步调优。 

**6.1.2 R-Net 、O-Net**

`R-Net`的网络结构与之前的P-Net 非常类似， P-Net的输入是`12×12×3`的图像，
R-Net是`24×24`心的图像，也就是说，R-Net 判断`24×24×3`的图像中是否含有人脸，
以及预测关键点的位置。R-Net的输出和P-Net完全一样，同样由人脸判别、 框回归、
 关键点位置预测三部分组成。
 
在实际应用中，对每个`P-Net`输出可能为人脸的区域都放缩到`24×24`的大小，再输入到
`R-Net`中，进行进一步判定。显然R-Net消除了P-Net中很多误判的情况。

进一步把所高得到的区域缩放成`48×48`的大小，输入到最后的`O-Net`中，O-Net的结构同
样与P-Net类似，不同点在于宫的输入是`48×48×3`的图像，网络的通道数和层数也更多了。

从`P-Net`到`R-Net`，最后再到`O-Net`，网络输入的图片越来越大，卷积层的通道数越
来越多，内部的层数也越来越多， 因此官们识别人脸的准确率应该是越来越高的。 同时
，`P-Net`的运行速度是最快的， `R-Net`的速度其次，`O-Net`的运行速度最慢。 之所
以要使用三个网络，是因为如果一开始直接对图中的每个区域使用`O-Net`， 速度会非常
慢。实际上`P-Net`先做了一遍过滤，将过滤后的结果再交给`R-Net`进行过滤，最后将过滤
后的结果交给效果最好但速度较慢的`O-Net`进行判别。这样在每一步都提前减少了需要判
别的数量，有效降低了处理时间。 
***
最后介绍`MTCNN`的`损失定义`和`训练过程`。 MTCNN中每个网络都有三部分输出，因此
损失也由三部分组成。 针对`人脸判别部分`，直接使用`交叉熵损失`，针对`框回归和
关键点判定`，直接使用`L2损失`。 最后这三部分损失各自乘以自身的权重再加起来，
就形成最后的总损失了。在`训练P-Net和R-Net`时，更关心`框位置的准确性`，而较少关注
`关键点判定的损失`，因此关键点判定损失的`权重很小`。对于`0-Net`，`关键点判定损失的
权重较大`。

## 6.2  使用深度卷积网络提取特征

**6.2.1 人脸的向量表示**

在理想的状况下，希望“向量表示”之间的距离可以直接反映人脸的相似度：
>* 对于同一个人的两张人脸图像，对应的向量之间的欧几里得距离应该比
较小。
>* 对于不同人的两张人脸图像3 对应的向量之间的欧几里得距离应该比较大。

例如，设人脸图像为x1,x2对应的特征为f(x1),f(x2），当x1,x2对应是同一个人
的人脸时，`f(x1)`,`f(x2）`的距离`||f(x1)-f(x2)||2`应该很小，而当`x1,x2`
是不同人的人脸时，`f(x1),f(x2）`的距离`||f(x1)-f(x2)||2` 应该很大。

**6.2.2 普通特征提取的不足**

在通常的图像应用中，可以去掉全连接层，使用卷积层的最后一层当作图像的“特征”
但如果对人脸识别问题同样采用这种方法，即便用卷积层最后一层估史为人脸的“向量表示”
，效果其实是不好的。

直撞使用`Softmax`训练得到的结果，它不符合希望特征真奇的特点：

>* 希望同一类对应的向量表示尽可能接近。 但这里同一类的点可能真奇很
大的类间距离。
>* 希望不同类对应的向量应该尽可能远。 但在国中靠中心的位置，各个类
别的距离都很近。

对于人脸图像同样会出现类似的情况。 对此，再很多改进方法。这里介绍其中两种
，一种是使用`三元组损失（Triplet Loss）`，一种是使用`中心损失`。

**6.2.3 三元组损失的定义**

`三元组损失（Triplet Loss）`的原理是：既然目标是特征之间的距离应当
具备某些性质，那么就围绕这个距离来设计损失。 具体地，每次都在训练数
据中取出三张人脸图像，第一张图像记为`x^a_i`， 第二张图像记为`x^p_i`，第三张国
像记为`x^n_i`。 在这样一个`“三元组”`中，`x^a_i`和`x^p_i`对应的是`同一个人的图像`
，而`x^m_i`是另外一个`不同的人的人脸图像`。因此，距离`||f(x^a_i)- f(x^p_i)||_2`
应该较小，而距离`||f(x^a_i)- f(x^n_i)||_2`应该较大。 

严格来说， 三元组损失要求下面的式子成立


```
||f(x^a_i)- f(x^p_i)||2^2 + α <  ||f(x^a_i)- f(x^n_i)||2^2
```

`即相同人脸间的距离平方至少要比不同人脸间的距离平方小α（取平方主要是方便求导）`

```
Li = [||f(x^a_i)- f(x^p_i)||2^2 + α - ||f(x^a_i)- f(x^n_i)||2^2]_+
```

这样的话， 当三元组的距离满足
`||f(x^a_i)- f(x^p_i)||2^2 + α <  ||f(x^a_i)- f(x^n_i)||2^2`
时，不产生任何损失，此时Li = 0 。当距离不满足上述等式时，就会有值为
`||f(x^a_i)- f(x^p_i)||2^2 + α - ||f(x^a_i)- f(x^n_i)||2^2`
的损失。此外，再训练时会固定||f(x)||2 = 1，以保证特征不会无限地“远离”

三元组损失直接对距离进行优化，因此可以解决人脸的特征表示问题。但是在训练过程
中，三元组的选择非常地高技巧性。如果每次都是随机选择三元组， 虽然模型可以正
确地收敛，但是并不能达到最好的性能。

**6.2.4 中心损失的定义**

与三元组损失不同，`中心损失（Center Loss）`不直接对距离进行优化，
`它保留了原有的分类模型，但又为每个类（在人脸模型中，一个类就对应一个人）指定了一个类别中心。
同一类的图像对应的特征都应该尽量靠近自己的类别中心，不同类的类别中心尽量远离。`
与三元组损失相比，使用中心损失训练人脸模型不需要使用特别的采样方法，而且利用较少
的图像就可以达到与三元组损失相似的效果。 下面就一起来学习中心损失的定义。

还是设输入的人脸图像为`xi`,该人脸对应的类别为`yi`，对每个类别都规定
一个类别中心，记作`c_yi`。 希望每个人脸图像对应的特征`f(xi）`都尽可能接
近真中心`c_yi` 。 因此定义中心损失为:

```
Li = 1/2(||f(xi) - c_yi||2^2)
```

多张图像的中心损失就是将它们的值加在一起

```
L_center = sum(Li)
```
这是一个非常简单的定义。 不过还高一个问题没有解决，那就是如何确定每个类别
的中心`c_yi`呢？从理论上来说，类别`yi`的最佳中心应该是最佳对应的所有图片的
特征的平均值。 但如果采取这样的定义，那么在每一次梯度下降时，都要对所有图片
计算一次`c_yi`，计算复杂度就太高了。 针对这种情况，不妨近似一处理下，在初始
阶段，先随机确定`c_yi`， 接着在每个batch内，使用`Li = 1/2(||f(xi) - c_yi||2^2)`
对当前batch内的`c_yi`也计算梯度，并使用该梯度更新`c_yi`。

此外，`不能只使用中心损失来训练分类模型`，还需要加入`Softmax`损失，也就是说，
最终的损失由两部分构成，即`L= Lsoftmax ＋λLcenter`，其中`λ`是一个超参数。

## 6.3 使用特征设计应用

在上一节中，当提取出特征后，剩下的问题就非常简单了。因为这种特征已经具有了相同
人对应的向量的距离小、不同人对应的向量距离大的特点。接下来，一般的应用有一下几类：

>* `人脸验证(Face Identification)`。就是检测A、B是否同属于同一个人。只需要
    计算向量之间的距离，设定合适的`报警阈值(threshold)`即可。
>* `人脸识别(Face Recognition)`。这个应用是最多的，给定一张图片，检测数据库
    中与之最相似的人脸。显然可以被转换为一个求距离的最邻近问题。
>* `人脸聚类(Face Clustering)`。在数据库中对人年进行聚类，直接用K-Means即可。


## 6.4 在Tenso「Flow中实现人脸识别

本节的程序来自于项目[https://github.com/davidsandberg/facenet](https://github.com/davidsandberg/facenet) 。

**6.4.1 项目环境设置**

参考6.4.1小节。

**6.4.2 LFW 人脸数据库**

在地址[http://vis-www.cs.umass.edu/lfw/lfw.tgz](http://vis-www.cs.umass.edu/lfw/lfw.tgz) 下载lfw数据集，并解压到~/datasets/中：
```
cd ~/datasets
mkdir -p lfw/raw
tar xvf ~/Downloads/lfw.tgz -C ./lfw/raw --strip-components=1
```

**6.4.3 LFW 数据库上的人脸检测和对齐**

对LFW进行人脸检测和对齐：

```
python src/align/align_dataset_mtcnn.py \
  ~/datasets/lfw/raw \
  ~/datasets/lfw/lfw_mtcnnpy_160 \
  --image_size 160 --margin 32 \
  --random_order
```

在输出目录~/datasets/lfw/lfw_mtcnnpy_160中可以找到检测、对齐后裁剪好的人脸。

**6.4.4 使用已有模型验证LFW 数据库准确率**

在百度网盘的chapter_6_data/目录或者地址https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk 下载解压得到4个模型文件夹，将它们拷贝到~/models/facenet/20170512-110547/中。

之后运行代码：
```
python src/validate_on_lfw.py \
  ~/datasets/lfw/lfw_mtcnnpy_160 \
  ~/models/facenet/20170512-110547/
```

即可验证该模型在已经裁剪好的lfw数据集上的准确率。

**6.4.5 在自己的数据上使用已有模型**

计算人脸两两之间的距离：
```
python src/compare.py \
  ~/models/facenet/20170512-110547/ \
  ./test_imgs/1.jpg ./test_imgs/2.jpg ./test_imgs/3.jpg
```

**6.4.6 重新训练新模型**

以CASIA-WebFace数据集为例，读者需自行申请该数据集，申请地址为http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html 。获得CASIA-WebFace 数据集后，将它解压到~/datasets/casia/raw 目录中。此时文件夹~/datasets/casia/raw/中的数据结构应该类似于：
```
0000045
  001.jpg
  002.jpg
  003.jpg
  ……
0000099
  001.jpg
  002.jpg
  003.jpg
  ……
……
```

先用MTCNN进行检测和对齐：
```
python src/align/align_dataset_mtcnn.py \
  ~/datasets/casia/raw/ \
  ~/datasets/casia/casia_maxpy_mtcnnpy_182 \
  --image_size 182 --margin 44
```

再进行训练：
```
python src/train_softmax.py \
  --logs_base_dir ~/logs/facenet/ \
  --models_base_dir ~/models/facenet/ \
  --data_dir ~/datasets/casia/casia_maxpy_mtcnnpy_182 \
  --image_size 160 \
  --model_def models.inception_resnet_v1 \
  --lfw_dir ~/datasets/lfw/lfw_mtcnnpy_160 \
  --optimizer RMSPROP \
  --learning_rate -1 \
  --max_nrof_epochs 80 \
  --keep_probability 0.8 \
  --random_crop --random_flip \
  --learning_rate_schedule_file
  data/learning_rate_schedule_classifier_casia.txt \
  --weight_decay 5e-5 \
  --center_loss_factor 1e-2 \
  --center_loss_alfa 0.9
```

打开TensorBoard的命令(<开始训练时间>需要进行替换)：
```
tensorboard --logdir ~/logs/facenet/<开始训练时间>/
```

#### 拓展阅读

- MTCNN是常用的人脸检测和人脸对齐模型，读者可以参考论文Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks 了解其细节。

- 训练人脸识别模型通常需要包含大量人脸图片的训练数据集，常用 的人脸数据集有CAISA-WebFace（http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html ）、VGG-Face（http://www.robots.ox.ac.uk/~vgg/data/vgg_face/ ）、MS-Celeb-1M（https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-millioncelebrities-real-world/ ）、MegaFace（ http://megaface.cs.washington.edu/ ）。更多数据集可以参考网站：http://www.face-rec.org/databases

- 关于Triplet Loss 的详细介绍，可以参考论文FaceNet: A Unified Embedding for Face Recognition and Clustering，关于Center Loss 的 详细介绍，可以参考论文A Discriminative Feature Learning Approach for Deep Face Recognition。
