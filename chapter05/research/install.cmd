protoc object_detection/protos/*.proto --python_out=.
SET PYTHONPATH=%cd%;%cd%\slim
echo success
@pause