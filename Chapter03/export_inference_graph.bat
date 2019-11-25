:: tensorflow 提供的的导出网络结构的脚本
python export_inference_graph.py ^
--alsologtostderr ^
--model_name=inception_v3 ^
--output_file=satellite/inception_v3_inf_graph.pb ^
--dataset_name=satellite

:: tensorflow 提供的保存训练模型参数的脚本
python freeze_graph.py ^
--input_graph=satellite/inception_v3_inf_graph.pb ^
--input_checkpoint=satellite/train_dir/model.ckpt-5271 ^
--input_binary=true ^
--output_node_names=InceptionV3/Predictions/Reshape_1 ^
--output_graph=satellite/frozen_graph.pb