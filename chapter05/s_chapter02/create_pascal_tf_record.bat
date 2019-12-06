:: workdir
cd D:\workspace_py\deepLearningDemo\chapter05\s_chapter02

:: gen pascal_train.record
python create_pascal_tf_record.py ^
--data_dir=VOCtrainval_11-May-2012/VOCdevkit/ ^
--year=VOC2012 ^
--set=train ^
--output_path=VOCtrainval_11-May-2012/pascal_train.record ^
--label_map_path=D:\workspace_py\deepLearningDemo\chapter05\research\object_detection\data\pascal_label_map.pbtxt

:: gen pascal_val.record
python create_pascal_tf_record.py ^
--data_dir=VOCtrainval_11-May-2012/VOCdevkit/ ^
--year=VOC2012 ^
--set=val ^
--output_path=VOCtrainval_11-May-2012/pascal_val.record ^
--label_map_path=D:\workspace_py\deepLearningDemo\chapter05\research\object_detection\data\pascal_label_map.pbtxt