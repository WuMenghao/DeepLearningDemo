# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

from chapter02.s_chap2 import cifar10
from chapter02.s_chap2 import cifar10_train

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './cifar10_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './cifar10_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")


def eval_once(saver, summary_writer, top_k_op, summary_op):
    """进行一次预测."""
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        # (1)恢复checkpoint, checkpoint 路径示例:
        # /project_path/cifar10_train/model.ckpt-0
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

    # (2)start_queue_runners
    coord = tf.train.Coordinator()  # 协调者
    threads = []
    try:
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                             start=True))
        num_iter = int(math.ceil(FLAGS.num_examples) / FLAGS.batch_size)
        true_count = 0  # 统计正确预测是数量
        total_sample_count = num_iter * FLAGS.batch_size
        step = 0
        while step < num_iter and not coord.should_stop():
            predictions = sess.run()
            true_count += np.sum(predictions)
            step += 1

        # (3)计算 precision 正确率
        precision = true_count / total_sample_count
        print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

        # (4)生成summary
        summary = tf.Summary()
        summary.ParseFromString(sess.run(summary_op))
        summary.value.add(tag='precision @ 1', simple_value=precision)
        summary_writer.add_summary(summary, global_step)
    except Exception as e:
        # 出现异常停止程序
        coord.request_stop(e)

    # (5)协调结束训练
    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """对训练好的CIFAR-10卷积神经网络进行一定步骤的评估"""
    with tf.Graph().as_default() as g:
        # (1)读取预测数据 images, labels
        eval_data = FLAGS.eval_data == 'test'
        images, labels = cifar10.inputs(eval_data=eval_data)
        # (2)构建同于计算logits的CIFAR-10预测神经网络模型
        logits = cifar10_train.interence(images)
        # (3)进行预测运算
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        # (4)恢复训练完成的模型变量的 moving average 用于进行预测评估
        variable_averages = tf.train.ExponentialMovingAverage(
            cifar10_train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)
        # (5)构建summary operation
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)
        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
