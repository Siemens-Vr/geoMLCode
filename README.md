# Geothermal Classification with PyTorch and MLflow

## Overview
This repository provides a PyTorch-based model to classify geothermal images into specific categories using transfer learning with ResNet18. The model integrates MLflow for comprehensive experiment tracking, logging, and reproducibility.

---

## Geothermal Model Deployment

### **Important Notice**
**This model is for demonstration and testing purposes only.** It is not intended for production or critical use. If your application requires high accuracy and reliability, consult verified models and resources.

### Model Download
Download the pretrained model from Hugging Face:  
**[Geothermal Model on Hugging Face](https://huggingface.co/Kamalikinuthia/Geothermal_model/commit/f281d69041cdcb8643c1cbf4c4d547802d325777)**
**Note**
Please refer to hugging face instructions on how to use Huggingface models

### Requirements
1. Clone this repository and download `server.py`.
2. Install the necessary packages:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Script**  
   It is recommended to use a virtual environment to avoid conflicts:
   ```bash
   # Create and activate a virtual environment
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`

   # Start the server
   python server.py
   ```

### **Disclaimer**
This image analysis model is intended for demonstration purposes and has limitations due to the training dataset size. Results should not be interpreted as professional advice. For critical decisions or precise analysis, please consult a qualified expert.

---

## Project Overview

### **Introduction**
This project classifies geothermal images by training a modified ResNet18 model on geothermal datasets. The model utilizes PyTorch, data augmentation, and cross-validation, while MLflow logs each experiment for tracking and reproducibility.

### **Model Architecture**
The model is based on ResNet18 pretrained on ImageNet. Key modifications include:

- **Fully Connected Layer**: 256 units
- **Activation**: ReLU
- **Dropout**: 0.5 probability
- **Output Layer**: Number of classes in the dataset

### **Dataset**
The Geothermal Dataset contains images from various geothermal categories and is available on Hugging Face:  
**[Geothermal Dataset on Hugging Face](https://huggingface.co/datasets/Kamalikinuthia/geothermal-dataset)**

- **Training Set**: Includes random resizing, horizontal flips, and rotation.
- **Validation Set**: Resized and normalized without augmentation.

### **Data Augmentation and Transformations**
Data augmentation enhances model generalization. Transformations applied:

- `RandomResizedCrop(size=224)`
- `RandomHorizontalFlip`
- `RandomRotation(15 degrees)`
- Normalization using ImageNet mean and standard deviation.

### **Training and Evaluation**
- **Loss Function**: Cross-entropy
- **Optimizer**: Adam
- **Learning Rate Scheduler**: ReduceLROnPlateau (reduces LR on validation loss plateau)
- **Metrics**: Accuracy and F1 score (tracked for training and validation sets)

### **Cross-Validation**
5-fold cross-validation is implemented with `KFold` from `sklearn.model_selection`. This setup trains the model on various dataset splits, logging accuracy and F1 score for each fold through MLflow.

---

