**Geothermal Classification with PyTorch and MLflow**

This repository contains a PyTorch-based implementation for classifying geothermal images. 
The model is trained on geothermal datasets using transfer learning with ResNet18, 
and MLflow is integrated for experiment tracking and logging.

**Introduction**
The goal of this project is to classify geothermal images into different categories based on image data. We employ a ResNet18 model, modified for classification,
and train it using PyTorch with data augmentation and cross-validation.
The results of each experiment are logged using MLflow for reproducibility and experiment tracking.

**Model Architecture**
The model used in this project is a ResNet18 architecture pretrained on ImageNet. The fully connected layer is replaced by a new head that consists of:

A fully connected layer (256 units)
ReLU activation
Dropout (0.5 probability)
A final classification layer corresponding to the number of classes

**Dataset**
The dataset used is the Geothermal Dataset, which contains images from different geothermal categories. It is preprocessed with the following transformations:
link ``https://huggingface.co/datasets/Kamalikinuthia/geothermal-dataset``
Random resizing, horizontal flips, and rotation for the training set
Normalization for both training and validation sets
Data Augmentation and Transformations
Data augmentation is applied to the training images to improve model generalization. The following transformations are used:

RandomResizedCrop (size 224)
RandomHorizontalFlip
RandomRotation (15 degrees)
Normalization (using ImageNet mean and standard deviation)
Validation images are resized and normalized but not augmented.

**Training and Evaluation**
Cross-entropy loss is used as the loss function.
The model is optimized with Adam optimizer.
ReduceLROnPlateau scheduler is employed to reduce the learning rate if validation loss plateaus.
Metrics such as accuracy and F1 score are tracked for both training and validation.
Cross-Validation
5-fold cross-validation is implemented using KFold from sklearn.model_selection. The model is trained on different splits of the dataset, and the performance metrics (accuracy and F1 score) are logged for each fold.
