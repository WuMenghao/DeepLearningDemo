:: 在对齐好的LFW数据库中验证已有模型的正确率：
SET PYTHONPATH=D:\workspace_py\deepLearningDemo\chapter06\facenet\src

python facenet/src/validate_on_lfw.py ^
datasets/lfw/lfw_mtcnnpy_160 ^
models/20170512-110547 ^
--lfw_pairs=facenet/data/pairs.txt ^
--gpu_memory_fraction=0.5