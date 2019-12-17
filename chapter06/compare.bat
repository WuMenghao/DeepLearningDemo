:: 计算人脸距离
SET PYTHONPATH=D:\workspace_py\deepLearningDemo\chapter06\facenet\src

python facenet/src/compare.py ^
models/20170512-110547 ^
test_imgs/1.jpg test_imgs/2.jpg test_imgs/3.jpg ^
--gpu_memory_fraction=0.5
