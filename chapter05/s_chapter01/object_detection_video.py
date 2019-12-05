# -*- coding: utf-8 -*-
# Created by: WU MENGHAO
# Created on: 2019/12/5

import numpy as np
import tensorflow as tf
import cv2
import os

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

MODEL_DIR = 'model/'
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_DIR + MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_DIR + MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('research/object_detection/data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

RESULT_DIR = 'result/'
DIST_FILE_NAME = 'output_video.mp4'
DATA_VIDEO_PATH = 'data/'


def show_image(image_np):
    cv2.imshow('dist', image_np)
    cv2.waitKey(0)


if __name__ == '__main__':
    # (1) 读取视频的一帧 获取帧的信息 开启输出流
    cap = cv2.VideoCapture('data/video1.mp4')
    ret, image_np = cap.read()
    height, weight = image_np.shape[:2]
    writer = cv2.VideoWriter(RESULT_DIR + DIST_FILE_NAME, -1,
                             cap.get(cv2.CAP_PROP_FPS), (image_np.shape[1], image_np.shape[0]))

    # (2) 加载model
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_grap_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            od_grap_def.ParseFromString(fid.read())
            tf.import_graph_def(od_grap_def, name='')

    # (3) 加载lab_map
    lab_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map=lab_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # (4) 对每帧进行目标检测，以视频方式写出
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')  # 为检测模型定义input 和 output tensors
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')  # 检测框
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')  # 得分
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')  # 分类
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')  # 标签
            while cap.isOpened():
                ret, image_np = cap.read()
                if len(np.array(image_np).shape) == 0:
                    break
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # 使用tensorflow进行计算
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # 可视化
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8
                )
                writer.write(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
                # show_image(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
