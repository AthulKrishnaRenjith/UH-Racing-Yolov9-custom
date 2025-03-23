# UH Racing Yolov9 Custom

This repository contains the custom implementation of the Yolov9 model for the UH Racing team. The project focuses on utilizing the Yolov9 model for object detection tasks specific to the requirements of racing.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Deployment on Jetson Nano](#deployment-on-jetson-nano)
- [Performance & Benchmarks](#performance--benchmarks)
- [Model Metrics](#model-metrics)

## Introduction

The UH Racing Yolov9 Custom project aims to provide an efficient and accurate object detection solution using the Yolov9 architecture. This project is tailored for the specific needs of the UH Racing team, enabling them to detect and classify objects relevant to their domain.

## Installation

To get started with this project, follow the steps below to set up the environment and install the necessary dependencies.

### Prerequisites
- Python 3.8+
- PyTorch with CUDA support (if using GPU)
- NVIDIA Jetson Nano (optional, for deployment)
- TensorRT (for optimized inference on Jetson Nano)

### Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/AthulKrishnaRenjith/UH-Racing-Yolov9-custom.git
    cd UH-Racing-Yolov9-custom
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To use the pretrained Yolov9 custom model for object detection, follow the steps below:

1. Ensure you have the necessary input data (images/videos) in the appropriate directory.
2. Run the detection script:
    ```bash
    python3 detect.py --weights runs/train/<experiment>/weights/best.pt --conf 0.1 --source <path_to_input_data> --device 0
    ```
3. The results will be saved in the `runs/detect` directory.

## Dataset

The dataset used for training the Yolov9 custom model comprises images and annotations specific to the Racing domain.

### Dataset Details
- Sources: Formula Student dataset (Roboflow)
- Annotation format: YOLOv9 format (txt)
- Preprocessing: Auto-orientation of pixel data (with EXIF-orientation stripping), Resize to 1920x1080 (Stretch)

Acknowledgement: The dataset used in this project is derived from the Formula Student dataset, which has been invaluable in training my object detection models.

![Dataset](/runs/train/exp1/labels.jpg)

## Training

To train the Yolov9 custom model, follow the steps below:

1. Prepare the dataset and update the configuration files as needed.
2. Run the training script:
    ```bash
	python3 -u train.py \
    	--batch <batch_size> --epochs <num_epochs> --img <image_size> --device <device_id> \
    	--min-items <min_items> --close-mosaic <close_mosaic_epoch> \
    	--data <data_config> \
    	--weights <pretrained_weights> \
    	--cfg <model_config> \
    	--hyp <hyperparameter_config>
    ```
3. Monitor the training process and evaluate the model's performance.

### Recommended Hyperparameters
- Batch size: 25 (adjust based on GPU memory)
- Learning rate: 0.01 (with warmup)
- Epochs: 80 (Adjust based on loss)
- Mixed precision training (FP16) supported

## Evaluation

To evaluate the trained Yolov9 custom model, use the evaluation script:
   ```bash
	   python3 val.py \
	   --task <task_type> \
	   --data <data_config> \
	   --batch <batch_size> \
	   --weights <model_weights>
   ```
The evaluation metrics will be displayed, and the results will be saved for further analysis.

## Deployment on Jetson Nano

To deploy the trained model on Jetson Nano with TensorRT optimization:

Convert the trained model to TensorRT format:
   ```bash
	   python3 export.py --weights runs/train/<experiment>/weights/best.pt --include engine --device 0 --half --simplify
   ```
## Performance & Benchmarks

| Model | Hardware | mAP (%) | FPS |
|--------|------------|---------|-----|
| YOLOv9 (PyTorch)  | NVIDIA A100        | 85.4 | 75 |
| YOLOv9 (ONNX)     | Jetson Nano (FP16) | 76.2 | (Untested) |
| YOLOv9 (TensorRT) | Jetson Nano (FP16) | 74.8 | 20 |

## Model Metrics

After training for 80 epochs, the model achieved the following performance:

![Metrics](/runs/train/exp1/results.png)



