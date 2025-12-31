[简体中文](README_CN.md) | [English](README.md)

# amlnn-model-zoo

# 简介

​		amlnn-model-zoo是基于amlnn toolkit完成模型转换与部署，实现主流常用算法的model zoo。demo包中提供完整的模型转换脚本，以及使用 Python API，OpenAI API 和 C API 对转换后的模型进行推理运行的完整流程。

**目的：** 帮助用户快速上手，完成算法模型在 Amlogic NPU平台上的部署。model zoo中丰富的算法库可以更好的指导客户AI产品落地。



# 依赖项

- 

# 支持列表

| Category               | Model_name                                                   | Dtype  | Model Link           | Platform      |
| ---------------------- | ------------------------------------------------------------ | ------ | -------------------- | ------------- |
| Classification         | [mobilenet_v2](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz) | INT8   | Link to server(TODO) | A311D2/S905X5 |
| Classification         | [resnet50-v2](https://github.com/onnx/models/blob/8e893eb39b131f6d3970be6ebd525327d3df34ea/vision/classification/resnet/model/resnet50-v2-7.onnx) | INT8   | Link to server(TODO) | A311D2/S905X5 |
| Object Detection       | [yolov8](https://github.com/ultralytics/ultralytics)         | INT8   | Link to server(TODO) | A311D2/S905X5 |
| Object Detection       | [yolov11](https://github.com/ultralytics/ultralytics)        | INT8   | Link to server(TODO) | A311D2/S905X5 |
| Object Detection       | [yoloworld](https://github.com/AILab-CVC/YOLO-World)         | INT8   | Link to server(TODO) | A311D2/S905X5 |
| Object Detection       | yoloe                                                        | INT8   | Link to server(TODO) | A311D2/S905X5 |
| Face Key Points        | [retinaface]([GitHub - biubug6/Pytorch_Retinaface: Retinaface get 80.99% in widerface hard val using mobilenet0.25.](https://github.com/biubug6/Pytorch_Retinaface)) | INT8   | Link to server(TODO) | A311D2/S905X5 |
| Text Detection         | ppocr-det                                                    | INT8   | Link to server(TODO) | A311D2/S905X5 |
| Pose Estimation        | blazepose_detect                                             | INT8   | Link to server(TODO) | A311D2/S905X5 |
| Pose Estimation        | blazepose_landmark                                           | INT8   | Link to server(TODO) | A311D2/S905X5 |
| Voiceprint recognition | [ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN)        | Hybrid | Link to server(TODO) | A311D2/S905X5 |
| Speech Recognition     | [whisper](https://github.com/openai/whisper)                 | Hybrid | Link to server(TODO) | A311D2/S905X5 |
| Image-Text Matching    | [clip](https://huggingface.co/openai/clip-vit-base-patch32)  | Hybrid | Link to server(TODO) | A311D2/S905X5 |
| Chat LLM               | deepseek                                                     | Hybrid | Link to server(TODO) | A311D2/S905X5 |



# Benchmark List(FPS)

| Examples           | Model_name   | input_shapes     | Dtype | S905X5 | A311D2 |
| ------------------ | ------------ | ---------------- | ----- | ------ | ------ |
| mobilenet          | mobilenet_v2 | [1, 3, 224, 224] | INT8  |        |        |
| resnet             | resnet50-v2  |                  |       |        |        |
| yolov8             | yolov8m      |                  |       |        |        |
| yolov11            |              |                  |       |        |        |
| yoloworld          | yoloworld    |                  |       |        |        |
| yoloe              | yoloe        |                  |       |        |        |
| retinaface         | retinaface   |                  |       |        |        |
| ppocr-det          |              |                  |       |        |        |
| blazepose_detect   |              |                  |       |        |        |
| blazepose_landmark |              |                  |       |        |        |
| ECAPA-TDNN         |              |                  |       |        |        |
| Whisper            | Whisper      |                  |       |        |        |



# Examples 编译





# **Release Notes**

| Version | Description   |
| ------- | ------------- |
| 1.0.0   | First Version |





​	