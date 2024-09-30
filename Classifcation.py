import os
import argparse
import mlflow
import mlflow.pytorch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from tqdm import tqdm

# Define argument parser for configuration
parser = argparse.ArgumentParser(description='Geothermal Classification Training')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--n_splits', type=int, default=5, help='number of folds for cross-validation')
parser.add_argument('--test_image', type=str, help='path to external image for testing')
args = parser.parse_args(['--batch_size', '32', 
                         '--epochs', '50', 
                         '--lr', '0.001', 
                         '--n_splits', '5'])

# Set up MLflow
mlflow.set_experiment("Geothermal Classification without Metadata")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transformations with data augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class GeothermalNet(nn.Module):
    def __init__(self, num_classes):
        super(GeothermalNet, self).__init__()
        self.resnet = models.resnet18(weights='DEFAULT')
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, image):
        return self.resnet(image)

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if img.mode=='RGBA':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)
        
        label = self.labels[idx]
        return img, label

def create_model(num_classes):
    return GeothermalNet(num_classes)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    best_val_loss = float('inf')
    patience = 10
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_preds, train_labels = [], []

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')
        
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')

        scheduler.step(val_loss)

        mlflow.log_metric("train_loss", epoch_loss, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("train_f1", train_f1, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)
        mlflow.log_metric("val_f1", val_f1, step=epoch)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print("Early stopping triggered")
            break

    return model

def load_model(model_path, num_classes):
    model = create_model(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0)

#function to test on external images(images not in the dataset)
# def test_external_image(model, image_path, device):
#     model.eval()
#     image = preprocess_image(image_path).to(device)
    
#     with torch.no_grad():
#         outputs = model(image)
#         _, predicted = torch.max(outputs, 1)
    
#     return predicted.item()

def main():
    # Load and prepare dataset
    try:
        dataset = load_dataset("Kamalikinuthia/geothermal-dataset")
        train_images = dataset['train']['image']
        train_labels = dataset['train']['label']
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)

    full_dataset = CustomDataset(images=train_images, labels=train_labels, transform=train_transform)

    # Cross-validation
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        print(f"Fold {fold+1}")
        
        with mlflow.start_run(run_name=f"fold_{fold+1}"):
            mlflow.log_params(vars(args))
            
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

            train_loader = DataLoader(full_dataset, batch_size=args.batch_size, sampler=train_subsampler)
            val_loader = DataLoader(full_dataset, batch_size=args.batch_size, sampler=val_subsampler)

            model = create_model(num_classes=len(set(train_labels))).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

            model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, args.epochs)

            # Test the model
            model.eval()
            test_preds, test_labels = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    test_preds.extend(preds.cpu().numpy())
                    test_labels.extend(labels.cpu().numpy())
            
            test_acc = accuracy_score(test_labels, test_preds)
            test_f1 = f1_score(test_labels, test_preds, average='weighted')

            mlflow.log_metric("test_acc", test_acc)
            mlflow.log_metric("test_f1", test_f1)

            print(f"Fold {fold+1} Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")

            # #  test with external image
            # if args.test_image:
            #     prediction = test_external_image(model, args.test_image, device)
            #     print(f"Prediction for external image: {prediction}")

if __name__ == "__main__":
    main()
