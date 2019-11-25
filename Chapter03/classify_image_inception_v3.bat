:: 对单张图片进行识别
python classify_image_inception_v3.py ^
--model_path=satellite/frozen_graph.pb ^
--label_path=data_prepare/pic/label.txt ^
--image_file=test_image.jpg