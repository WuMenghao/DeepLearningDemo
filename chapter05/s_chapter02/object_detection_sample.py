# -*- coding: utf-8 -*-
# Created by: WU MENGHAO
# Created on: 2019/12/4

import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import pylab
import matplotlib
import scipy.misc

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'VOCtrainval_11-May-2012/export/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('VOCtrainval_11-May-2012/', 'pascal_label_map.pbtxt')
NUM_CLASSES = 90

# For the sake of simplicity we will use only 2 images:
PATH_TO_TEST_IMAGE_DIR = 'test/'
RESULT_DIR = 'result/'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGE_DIR, file_name) for file_name in os.listdir(PATH_TO_TEST_IMAGE_DIR)]

IMAGE_SIZE = (12, 8)


def show_image(image_np):
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
    pylab.show()


def save_image(name, image):
    scipy.misc.imsave(name, image)


def load_image_array(image_path):
    """
    将image 转换为nparray
    :param image_path:
    :return:
    """
    return np.uint8(scipy.misc.imread(image_path))


def load_model():
    """
    加载 model 到 tensorflow
    :return:
    """
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


def load_label_map():
    """
    加载 label map
    :return:
    """
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index


def render_detection(graph, sess, category_index):
    """
    进行物体检测
    :param graph:
    :param sess:
    :param category_index:
    :return:
    """
    image_tensor = graph.get_tensor_by_name('image_tensor:0')  # 为检测模型定义input 和 output tensors
    detection_boxes = graph.get_tensor_by_name('detection_boxes:0')  # 检测框
    detection_scores = graph.get_tensor_by_name('detection_scores:0')  # 得分
    detection_classes = graph.get_tensor_by_name('detection_classes:0')  # 分类
    num_detections = graph.get_tensor_by_name('num_detections:0')  # 标签
    for i in range(0, len(TEST_IMAGE_PATHS)):
        image_path = TEST_IMAGE_PATHS[i]
        # (1)读取图片
        image_np = load_image_array(image_path)
        # show_image(image_np)
        # (2)扩维到[1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # (3)实际进行物体检测
        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded}
        )
        # (4)结果可视化
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        # show_image(image_np)
        save_image(RESULT_DIR + 'image%s.jpg' % i, image_np)


def main(_):
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False,
                                    # gpu_options=gpu_options,
                                    device_count={'GPU': 1})
    with tf.Graph().as_default() as detection_graph:
        with tf.Session(graph=detection_graph,config=session_config).as_default() as sess:
            load_model()
            category_index = load_label_map()
            render_detection(detection_graph, sess, category_index)


if __name__ == '__main__':
    tf.app.run()
