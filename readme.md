
torch-model-archiver --model-name yolov4 --version 1.0 --handler yolov4/handler.py --serialized-file yolov4/dahua_v4.pb --extra-files yolov4/numpy_eval.py --export-path .

torchserve --start --ts-config config.properties --model-store . --models yolov4.mar



