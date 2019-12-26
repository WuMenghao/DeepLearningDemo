:: 计算人脸距离
SET PYTHONPATH=D:\workspace_py\deepLearningDemo\chapter06\facenet\src

python facenet/src/compare.py ^
models/20191224-152347 ^
test_imgs03/001.png test_imgs03/002.png test_imgs03/003.png test_imgs03/004.png test_imgs03/005.png test_imgs03/006.png ^
test_imgs03/007.png test_imgs03/008.png test_imgs03/009.png test_imgs03/010.png test_imgs03/011.png test_imgs03/012.png ^
--gpu_memory_fraction=0.0
