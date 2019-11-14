# -*- coding: utf-8 -*-
import datetime
import time
import tensorflow as tf
from chapter02.s_chap2 import cifar10

FLAGS = cifar10.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_string('train_dir', './cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")

NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000

NUM_EPOCHS_PER_DECAY = 350.0  # 减小学习率后的epochs
INITIAL_LEARNING_RATE = 0.1  # 初始学习率
LEARNING_RATE_DECAY_FACTOR = 0.1  # 学习率下降因子
MOVING_AVERAGE_DECAY = 0.9999  # moving average下降因子


# 函数的输入参数为images,即是图像的Tensor
# 输出是图像属于各个类别的Logit
def interence(image):
    # 建立第一层卷积层
    with tf.variable_scope('conv1') as scope:
        # 卷积核
        kernel = cifar10._variable_with_weight_decay('weights',
                                                     shape=[5, 5, 3, 64],
                                                     stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(image, kernel, [1, 1, 1, 1], padding='SAME')
        biases = cifar10._variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        # summary是将输出报告到TensorBoard
        cifar10._activation_summary(conv1)

    # 第一层卷积层的池化
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

    # 局部影响归一化(LRN)，现在大多数模型不采用
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    # 第二层卷积
    with tf.variable_scope('conv2') as scope:
        kernel = cifar10._variable_with_weight_decay('weights',
                                                     shape=[5, 5, 64, 64],
                                                     stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = cifar10._variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        cifar10._activation_summary(conv2)

    # LRN层
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')

    # 第二层卷积层的池化
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # 全连接层1
    with tf.variable_scope('local3') as scope:
        # 后面不再做卷积了,所以把pool2进行reshape,方便做全连接
        reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = cifar10._variable_with_weight_decay('weights', shape=[dim, 384],
                                                      stddev=0.04, wd=0.004)
        biases = cifar10._variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        # 全连接 输出是relu(Wx+b)
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases,
                            name=scope.name)
        cifar10._activation_summary(local3)

    # 全连接层2
    with tf.variable_scope('local4') as scope:
        weights = cifar10._variable_with_weight_decay('weights', shape=[384, 192],
                                                      stddev=0.04, wd=0.004)
        biases = cifar10._variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases,
                            name=scope.name)
        cifar10._activation_summary(local3)

    # 全连接+Softmax分类
    # 这里不显示进行Softmax变换,只输出变换前的Logit(即变量softmax_linear)
    with tf.variable_scope('softmax_linear') as scope:
        weights = cifar10._variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                                      stddev=1 / 192.0, wd=0.0)
        biases = cifar10._variable_on_cpu('biases', [NUM_CLASSES],
                                          tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        cifar10._activation_summary(softmax_linear)

    return softmax_linear


# 计算交叉熵损失
def _loss(logits, labels):
    # cost labels type
    labels = tf.cast(labels, tf.int64)
    # calculate cross_entropy
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    # reduce cross_entropy_mean
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    # add to losses
    tf.add_to_collection('losses', cross_entropy_mean)
    # calculate total_loss
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


# 计算交叉熵损失均值并关联summary
def _add_loss_summaries(total_loss):
    # 计算所有loss 和 total_lost 的 moving_average
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    # 绑定scalar summary
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op


def _train(total_loss, global_step):
    # (1)定义学习率相关的参数
    num_batch_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batch_per_epoch * NUM_EPOCHS_PER_DECAY)
    # (2)使用exponential_decay函数控制学习率，以学习步长值为基础
    lr = tf.train.exponential_decay(learning_rate=INITIAL_LEARNING_RATE,
                                    global_step=global_step,
                                    decay_steps=decay_steps,
                                    decay_rate=LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)
    # (3)计算交叉熵损失均值并关联summary
    loss_averages_op = _add_loss_summaries(total_loss)
    # (4)计算梯度
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
    # (5)应用梯度
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    # (6)为训练变量添加直方图
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
    # (7)为梯度添加直方图
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    # (8)追踪所有可训练变量的moving averages
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # 顺序控制
    with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


# 训练控制
def train():
    with tf.Graph().as_default():
        # 定义global_step
        global_step = tf.contrib.framework.get_or_create_global_step()
        # 获取images
        images, labels = cifar10.distorted_inputs()
        # 构建神经网络
        logits = interence(images)
        # 计算交叉熵损失
        loss = _loss(logits, labels)
        # 训练以更新模型
        train_op = _train(loss, global_step)

        # 日志
        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime"""

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f'
                                  'sec/batch)')
                    print(
                        format_str % (datetime.datetime.now(), self._step,loss_value,
                                      examples_per_sec, sec_per_batch))

        # 进行监督学习
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.train_dir,
                hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                       tf.train.NanTensorHook(loss),
                       _LoggerHook()],
                config=tf.ConfigProto(
                    log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)

# tf主函数
def main(arg=None):
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
    tf.app.run()