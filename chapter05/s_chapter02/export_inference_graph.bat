:: 如何将train_dir中的checkpoint文件导出并用于单张图片的目标检测？
:: TensorFlow Object Detection API提供了一个export_inference_graph.py脚本
:: 用于导出训练好的模型。具体方法是执行．
python D:\workspace_py\deepLearningDemo\chapter05\research\object_detection\export_inference_graph.py ^
--input_type_image_tensor ^
--pipeline_config_path=VOCtrainval_11-May-2012/voc.config ^
--trained_checkpoint_prefix=VOCtrainval_11-May-2012/train_dir/model.ckpt-2613 ^
--output_directory=VOCtrainval_11-May-2012/export