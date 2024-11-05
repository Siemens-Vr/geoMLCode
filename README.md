**Geothermal Classification with PyTorch and MLflow**

**Geothermal Model Deployment**
**Important**
**Please Note:**
This model is intended for demonstration and testing purposes only. Downloading and deploying this model is not recommended for production environments or critical applications.

Model Download
Download the model file from the following link:
Geothermal Model on Hugging Face
`https://huggingface.co/Kamalikinuthia/Geothermal_model/commit/f281d69041cdcb8643c1cbf4c4d547802d325777`
Requirements
Clone this repository and download server.py.
Install the required Python packages by running:
bash
Copy code
pip install -r requirements.txt
Run the Script:
It’s recommended to use a virtual environment to avoid conflicts with global packages. To activate the virtual environment and run the script:
bash
Copy code
# Create and activate a virtual environment
python -m venv env
source env/bin/activate   # On Windows, use `env\Scripts\activate`

# Run the server
python server.py
**Disclaimer
This image analysis model is designed for demonstration purposes and may have limitations due to the dataset size used in its training. The results provided should not be interpreted as professional or conclusive advice. For critical decisions or precise analysis, please understand that you have to consult a qualified professional or specialist in the field.**



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
