# EfficientTeacher: Semi-Supervised Object Detection for YOLOv5

EfficientTeacher is a semi-supervised object detection algorithm based on YOLOv5, designed to improve detection performance by utilizing both labeled and unlabeled data. This approach helps train a more robust model with fewer labeled samples, making it suitable for real-world use cases where large labeled datasets are difficult to obtain.

This version of EfficientTeacher has been slightly modified to streamline the training and conversion process. Follow the instructions below to train YOLOv5, convert the model, and begin semi-supervised training.

## Setup

Before getting started, ensure you have the necessary dependencies installed and the appropriate dataset prepared, similar to the original EfficientTeacher repository.

1. Clone the repository and install the necessary libraries:
    ```bash
    git clone https://github.com/your_repo/efficientteacher
    cd efficientteacher
    pip install -r requirements.txt
    ```

2. Organize the dataset as follows:
    ```
    efficientteacher/
      ├── data/
      └── datasets/
          └── coco/
              └── images/
              └── labels/
    ```

3. Download the dataset:
    ```bash
    bash data/get_coco.sh
    ```

4. Download the training and validation image lists:
    ```bash
    bash data/get_label.sh
    ```

## Training Steps

### Step 1: Train YOLOv5

The first step is to train the YOLOv5 model using your dataset. 

### Step 2: Convert the YOLOv5 Model to EfficientTeacher Format

After training YOLOv5, convert the model to the EfficientTeacher format using the provided conversion script:

```bash
python scripts/convert_pt_to_efficient.py --weights <path_to_yolov5_model.pt> --cfg configs/sup/custom/yolov5l_custom.yaml
```

This step transforms the YOLOv5 model into a format compatible with EfficientTeacher’s semi-supervised training framework.

### Step 3: Validate the YOLOv5 Model

Before proceeding to semi-supervised training, validate the converted model to ensure that it works correctly within the EfficientTeacher framework. Run the following command:

```bash
python val.py --cfg configs/sup/custom/yolov5l_custom.yaml --weights efficient-yolov5l.pt
```

This command validates the YOLOv5 model on your test data and outputs the evaluation metrics.

### Step 4: Start Semi-Supervised Training

Once the model is validated, begin semi-supervised training using both labeled and unlabeled data:

```bash
export CUDA_VISIBLE_DEVICES="0"
torchrun --nproc_per_node=1 --master_addr=127.0.0.2 --master_port=29502 train.py --cfg configs/sup/custom/yolov5l_custom.yaml
```

This command starts the semi-supervised training process, leveraging the EfficientTeacher framework to improve the model's performance using unlabeled data.


## License

EfficientTeacher is released under the GPL-3.0 License. For more details, please see the `LICENSE` file.

