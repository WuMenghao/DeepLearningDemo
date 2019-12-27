:: 计算人脸距离
SET PYTHONPATH=D:\workspace_py\deepLearningDemo\chapter06\facenet\src

python facenet/src/compare.py ^
models/20191224-152347 ^
test_imgs03/001.png test_imgs03/002.png test_imgs03/003.png ^
--gpu_memory_fraction=0.0
