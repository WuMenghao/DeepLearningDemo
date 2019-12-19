:: 对CASIA数据库进行人年检测和对齐
SET PYTHONPATH=D:\workspace_py\deepLearningDemo\chapter06\facenet\src

python facenet/src/align/align_dataset_mtcnn.py ^
datasets/casia/raw ^
datasets/casia/casia_maxpy_mtcnnpy_182 ^
--gpu_memory_fraction=0.5 ^
--image_size=182 ^
--margin=44