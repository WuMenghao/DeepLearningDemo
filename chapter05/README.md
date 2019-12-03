## 5.1 深度学习中目标检测的原理
#### 5.1.1  R-CNN的原理
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
R-CNN的全称是Region-CN1叶， 它可以说是第一个成功地将深度学习应
用到目标检测上的算法。 后面将要学习的Fast R-CNN、 Faster R-CNN全部
都是建立在R-CNN基础上的。 
 
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
传统的目标检测方法大多以图像识别为基础。一般可以在图片上使用穷
举法选出所高物体可能出现的区域框，对这些区域框提取特征并使用图像识
别万法分类，得到所高分类成功的区域后， 通过非极大值抑制（Non-maximum 
suppression ）输出结果。 
 
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
R-C阳遵循传统目标检测的思路， 同样采用提取框、 对每个框提取特
征、 图像分类、非极大值抑制四个步骤进行目标检测。 只不过在提取特征这
一步，将传统的特征（如SIFT、 HOG特征等）；奥成了深度卷积网络提取的
特征。 R-CNN的整体框架如图5-2所示。

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
对于原始图像， 首先使用Selective Search搜寻可能存在物体的区域。
Selective Search可以从图像中启发式地搜索出可能包含物体的区域。相比穷
举而言，SelectiveSearch可以减少一部分计算量。 下一步，将取出的可能含
高物体的区域送入CNN中提取特征。 CNN通常是接受一个固定大小的国
像，而取出的区域大小却各高不同。 对此，R-CN1司的做法是将区域缩放到
统一大小， 再使用CN1司提取特征。 提取出特征后使用SVM进行分类3 最
后通过非极大值抑制输出结果。

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
R-CNN的训练、可以分成下面四步：
> * 在数据集上训练CNN。 R-CNN论文中使用的CNN网络是AlexNet1 ，数
 据集为ImageNet。
> * 在目标检测的数据集上，对训练好的CNN做微调。
> * 用Selective Search搜索候选区域，统一使用微调后的CNl'叶对这些区域
提取特征，并将提取到的特征存储起来。
> * 使用存储起来的特征，训练SVM分类器。

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
尽管R-CNN的识别框架与传统方法区别不是很大，但是得益于CNN优
异的特征提取能力，R-CNN的效果还是比传统方法好很多。 如在VOC2001 
数据集上，传统方法最高的平均精确度mAP( mean Average Precision )为
40%左右，而R-CNN的mAP达到了58.5%!

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
R-CNN的缺点是计算量太大。 在一张图片中，通过Selective Search得
到的有效区域往往在1000个以上，这意昧着要重复计算1000多次神经网络，
非常耗时。 另外，在训练、阶段，还需要把所高特征保存起来，再通过SVM
进行训练，这也是非常耗时且麻烦的。下面将要介绍的Fast R-CNN和Faster
R-CNN在一定程度上改进了R-CNN计算量大的缺点，不仅速度变快不少，
识别准确率也得到了提高。

#### 5.1.2  SPPNet的原理
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
在学习R-CNN的改进版FastR-CNN之前，作为前置知识，再必要学习
SPPNet的原理。 SPPNet的英文全称是SpatialPyramid Pooling Convolutional 
Networks , 翻译成中文是“空间金字塔池化卷积网络”。 昕起来十分高深，
实际上原理并不难，简单来讲，SPPNet主要做了一件事情：将CNN的输入从固
定尺寸改进为任意尺寸。 例如，在普通的CNN结构中，输入图像的尺
寸往往是固定的（如224x224像素），输出可以看做是一个固定维数的向量。
SPPNet 在普通的CNN结构中加入了ROI池化层（ROI Pooling ），使得网络
的输入图像可以是任意尺寸的，输出则不变，同样是一个固定维数的向量。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
ROI 池化层一般跟在卷积层后面，白的输入是任意大小的卷积，输出是
固定维数的向量， 如国5-3所示。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
为了说清楚为什么ROI池化层能够把任意大小的卷积特征转换成固定
~度的向量，不妨设卷积层输出的宽度为w， 高度为h，通道为c。不管输
入的图像尺寸是多少，卷积层的通道数都不会变，也就是说c是一个常数。
而w、 h会随着输入图像尺寸的变化而变化，可以看作是两个变量。以上图
中的ROI池化层为例3 百首先把卷积层划分为4畔的网格，每个网恪的竟
是w/4、 高是h/ 4、通道数为c。 当不能整除时，需要取整。接着，对每个
网格中的每个通道，都取出真最大值，换句话说，就是对每个网格内的特征
做最大值池化(Max Pooling)。这个4x4的网格最终就形成了16c维的特征。
接着，再把网络划分成2x2的网恪，用同样的方法提取特征，提取的特征的
长度为4c。再把网络划分为1x1 的网恪，提取的特征的长度就是c，最后的
lxl的划分实际是取出卷积中每个通道的最大值。最后，将得到的特征拼接
起来，得到的特征是16c+4c+c=21c维的特征。 很显然，这个输出特征的长
度与w、 h两个值是无关的，因此ROI池化层可以把任意宽度、高度的卷积
特征转换为固定长度的向量。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
应该怎么把ROI 池化层用到目标检测中来呢？真实，可以这样考虑该
问题．网络的输入是一张图像，中间经过若干卷积形成了卷积特征，这个卷
积特征实际上和原始图像在位置上是高一定对应关系的。 如图5-4所示， 原
始图像中有一辆汽车，2使得卷积特征在同样位置产生了激活。 因此，原始
图像中的候选框，实际上也可以对应到卷积特征中相同位置的框。由于候选
框的大小千变万化，对应到卷积特征的区域形状也各高不同，但是不用担心，
利用ROI池化层可以把卷积特征中的不同形状的区域对应到同样长度的向
量特征。 综合上述步骤，就可以将原始图像中的不同长宽的区域都对应到一
个固定长度的向量特征，这就完成了各个区域的特征提取工作。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
在R-CNN中，对于原始图像的各种候选区域框，必须把框中的图像缩
放到统一大小，再对每一张缩放后的图片提取特征。 使用ROI 池化层后，
就可以先对图像进行一遍卷积计算，得到整个图像的卷积特征；接着，对于
原始图像中的各种候选框，只需要在卷积特征中找到对应的位置框，再使用
ROI池化层对位置框中的卷积提取特征，就可以完成特征提取工作。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
R-CNN和SPPNet的不同点在于，R-CN1吁要对每个区域计算卷积3 而
SPPNet 只需要计算一次，因此SPPNet的效率比R-CNN高得多。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
R-CN1付日SPPNet的相同点在于，它们都遵循着提取候选框、提取特征、
分类几个步骤。在提取特征后，官们都使用了SVM进行分类。

#### 5.1.3  Fast R-CNN的原理
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
在SPPNet中，实际上特征提取和区域分类两个步骤还是分离的。 只是
使用ROI池化层提取了每个区域的特征，在对这些区域分类时，还是使用
传统的SVM作为分类器。FastR-CNN相比SPPNet更进一步，不再使用SVM
作为分类器，而是使用神经网络进行分类，这样就可以同时训练特征提取网
络和分类网络，从而取得比SPPNet更高的准确度。FastR-CNN的网络结构
如图5-5所示。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
对于原始图片中的候选框区域，和SPPNet中的做法一样，都是将包映
射到卷积特征的对应区域（即国5-5中的ROIprojection ），然后使用ROI池
化层对该区域提取特征。 在这之后，SPPNet是使用SVM对特征进行分类，
而Fast R-CNN则是直接使用全连接层。全连接层高两个输出，一个输出负
责分类（即图5-5中的Softmax），另一个输出负责框回归（即图5-5中的bbox regressor）。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
先说分类，假设要在图像中检测K类物体，那么最终的输出应该是K+I
个数，每个数都代表该区域为某个类别的概率。 之所以是K+I 个输出而不
是K个输出，是因为还需要一类“背景类”，针对该区域无目标物体的情况。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
再说框回归，框回归实际上要做的是对原始的检测框进行某种程度的
“校准”。因为使用Selective Search 获得的框高时存在一定偏差。设通过
Selective Search得到的框的四个参数为`(x,y,w,h)`，其中`(x,y)`表示框左上
角的坐标位置，`(w,h)`表示框的宽度和高度。 而真正的框的位置用
`(x',y',w',h')` 表示，框回归就是要学习参数`((x'-x)/w,(y'-y)/w,ln(w'/w),ln(h'/h)`
，其中`(x'-x)/w`、`(y'-y)/w`两个数表示与尺度无关的平移量，而`ln(w'/w)`、`ln(h'/h)`
两个数表示的是和尺度无关的缩放量。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Fast R-CNN与SPPNet最大的区别就在于，FastR-CNN不再使用SVM
进行分类，而是使用一个网络同时完成了提取特征、判断类别、 框回归三项
工作。

#### 5.1.4  Faster R-CNN的原理
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Fast R-CNN看似很完美了，但在FastR-CNN中还存在着一个高点尴尬的问题：他需
要先使用`SelectiveSearch`提取框，这个方法比较慢，再时，检测一张图片，大部
分时间不是花在计算神经网络分类上，而是花在`SelectiveSearch`提取框上!在
Fast R-CNN升级版Faster R-CNN中，用`RPN`网络`(Region Proposal Network)`取代
了SelectiveSearch ，不仅速度得到大大提高，而且还获得了更加精确的结果。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
`RPN`还是需要先使用一个CNN网络对原始图片提取特征。为了方便读
者理解，不妨设这个前置的CNN提取的特征为`5l×39×256`，即高为`51`、 宽
为`39`、通道数为`256`。对这个卷积特征再进行一次卷积计算，保持宽、 高、
通道不变，再次得到一个`5l×39×256`的特征。 为了方便叙述， 先来定义一个
“位置”的概念：对于一个`51×39×256`的卷积特征，称它一共有`51×39`个
“位置”。让新的卷积特征的每一个"位置"都"负责"原图中对应位置9种尺寸的
框的检测，检测的目标是判断框中是否存在一个物体’因此共高`5l×39×9`个
“框”。在FasterR-CNN的原论文中，将这些框都统一称为`“anchor”`。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
`anchor`的9种尺寸如图5-7所示，它们的面积分别`128^2` `256^2`  `512^2`。
每种面积又分为3 种长竟比，分别是`2 :  1` 、 `1 : 2`、 `1 :  1` 。 `anchor`
的尺寸实际是属于可调的参数，不同任务可以选择不同的尺寸。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
对于这`5l×39`个位置和`5l×39×9`个`anchor`，图5-8展示了接下来每个位
置的计算步骤。设`k`为单个位置对应的`anchor`的个数，此时`k=9`。 首先使用
一个`3×3`的渭动窗口，将每个位置转攘为一个统一的`256维的特征`，这个恃
征对应了两部分的输出。 一部分表示该位置的`anchor`为物体的概率，这部
分的总输出长度为`2×k`（一个`anchor`对应两个输出:是物体的概率＋不是物
体的概率）。 另一部分为框回归，框回归的含义与FastR-CM词中一样，一个
`anchor`对应`4个框回归参数`，因此框回归部分的总输出的长度为`4*k`。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Faster R-CNN使用即N生成候选框后3 剩下的网络结构和FastR-CNN 
中的结构一模一样。在训练过程中，需要训练两个网络，一个是RPN网络，
一个是在得到框之后使用的分类网络。 通常的做法是交替训练，即在一个
batch内，先训练RPN网络一次，再训练分类网络一次。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<table>
 <caption align="top">表5-1 R-CNN、 FastR-CNN、 FasterR-CNN的对比</caption> 
<thead>
    <td>项 目</td><td>R-CNN</td><td>Fast R-CNN</td><td>Faster R-CNN</td>
</thead>
<tbody>
    <tr>
        <td>提取候选框 </td><td>Selective Search </td><td>Selective Search </td><td>RPN 网络</td>
    </tr>
    <tr>
        <td>提取特征 </td><td>卷积神经网络(CNN)</td><td colspan="2" rowspan="2">卷积神经网络＋ROI池化</td>
    </tr>
    <tr>
        <td>特征分类 </td><td>支持向量机(SVM)</td>
    </tr>
</tbody>
</table>