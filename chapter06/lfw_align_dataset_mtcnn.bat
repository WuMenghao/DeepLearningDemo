:: 对LFW数据库进行人年检测和对齐
SET PYTHONPATH=D:\workspace_py\deepLearningDemo\chapter06\facenet\src

python facenet/src/align/align_dataset_mtcnn.py ^
datasets/lfw/raw ^
datasets/lfw/lfw_mtcnnpy_160 ^
--gpu_memory_fraction=0.5 ^
--image_size=160 ^
--margin=32 ^
--random_order