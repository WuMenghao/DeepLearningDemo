:: 数据准备 生成.tfrecord文件
python data_convert.py -t pic/ ^
--train-shards 2 --validation-shards 2 ^
--num-threads 2 --dataset-name satellite