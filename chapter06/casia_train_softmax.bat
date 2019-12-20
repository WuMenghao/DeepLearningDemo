:: 使用casia人年识别库训练MTCNN模型
set PYTHONPATH = =D:\workspace_py\deepLearningDemo\chapter06\facenet\src

python facenet/src/train_softmax.py ^
--logs_base_dir=log/ ^
--models_base_dir=models/20170512-110547/ ^
--data_dir=datasets/casia/casia_maxpy_mtcnnpy_182 ^
--image_size=160 ^
--model_def=models.inception_resnet_v1 ^
--lfw_dir=datasets/lfw/lfw_mtcnnpy_160 ^
--optimizer=RMSPROP ^
--learning_rate=-1 ^
--max_nrof_epoch=80 ^
--keep_probability=0.8 ^
--random_crop ^
--random_flip ^
--learning_rate_schedule_file=facenet/data/learning_rate_schedule_classifier_casia.txt ^
--weight_decay=5e-5 ^
--center_loss_factor=1e-2 ^
--center_loss_alfa=0.9 ^
--gpu_memory_fraction=0 ^
--lfw_pairs=facenet/data/pairs.txt
