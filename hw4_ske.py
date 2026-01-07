import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import argparse
import numpy as np
import glob
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from time import time
import torchvision

class SUN397Dataset(Dataset):
    """
    A custom dataset class for loading the SUN397 dataset.
    """

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # ONLY these 4 classes are valid
        valid_classes = ['airport_terminal', 'bedroom', 'dining_room', 'living_room']
        
        # get all the class folders
        class_names = []
        
        # Try structure with letter subdirectories first (a/airport_terminal, b/bedroom, etc.)
        letters_dirs = ['a', 'b', 'd', 'l']
        found_with_letters = False
        
        for letter in letters_dirs:
            letter_path = os.path.join(data_dir, letter)
            if os.path.exists(letter_path):
                # Get class folders within this letter directory
                for class_name in os.listdir(letter_path):
                    if class_name in valid_classes:  # ONLY valid classes
                        class_path = os.path.join(letter_path, class_name)
                        if os.path.isdir(class_path):
                            class_names.append((class_name, class_path))
                            found_with_letters = True
        
        # If no letter subdirectories found, look directly in data_dir
        if not found_with_letters:
            for class_name in valid_classes:  # ONLY check valid classes
                class_path = os.path.join(data_dir, class_name)
                if os.path.exists(class_path) and os.path.isdir(class_path):
                    class_names.append((class_name, class_path))

        # Sort class names to ensure consistent label mapping
        class_names.sort(key=lambda x: x[0])

        # map the index with the label 
        self.class_to_label = {class_name: idx for idx, (class_name, _) in enumerate(class_names)}
    
        # Load all image paths and labels
        for class_name, class_path in class_names:
            label = self.class_to_label[class_name]
        
            # Get all image files in this class folder
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg')):
                    img_path = os.path.join(class_path, img_file)
                    self.image_paths.append(img_path)
                    self.labels.append(label)

        print(f"Loaded {len(self.image_paths)} images from {len(self.class_to_label)} classes")
        print(f"Classes: {self.class_to_label}")

    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    

class CNN(nn.Module):
    """
    Balanced 5-Layer CNN
    Architecture: 32 -> 64 -> 128 -> 256 -> 512 channels
    Trained from scratch to achieve 75%+ accuracy
    """
    def __init__(self, num_classes=4):
        super(CNN, self).__init__()
        
        # --- BLOCK 1: 224 -> 112 ---
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- BLOCK 2: 112 -> 56 ---
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- BLOCK 3: 56 -> 28 ---
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- BLOCK 4: 28 -> 14 ---
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- BLOCK 5: 14 -> 7 ---
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Feature map size: 512 * 7 * 7 = 25,088
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(512 * 7 * 7, 512)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Conv blocks with BatchNorm -> ReLU -> Pool
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # FC layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    
    
def calculate_mean_std(**kwargs):
    data_dir = kwargs.get('data_dir', './data')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor() 
    ])
    
    # Recursively find all .jpg files
    image_paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith('.jpg'):
                image_paths.append(os.path.join(root, file))
    
    if len(image_paths) == 0:
        return [0.5, 0.5, 0.5], [0.2, 0.2, 0.2]
    
    mean = torch.zeros(3)
    std_sum = torch.zeros(3)
    total_images = 0
    
    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert('RGB')
            image = transform(image)
            
            mean += image.view(3, -1).mean(1)
            std_sum += image.view(3, -1).std(1)
            total_images += 1
        except:
            continue
    
    mean = mean / total_images
    std = std_sum / total_images
    
    # Return as plain Python floats
    return [float(mean[0]), float(mean[1]), float(mean[2])], [float(std[0]), float(std[1]), float(std[2])]

def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    runloss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        runloss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = runloss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def test(model, test_loader, criterion, device):
    model.eval()
    testloss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            testloss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = testloss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                        default='./data',
                        help='Path to data directory')
    
    parser.add_argument('--train_dir', type=str, 
                        default='welcome/to/CNN/homework',
                        help='Path to training data directory')
    
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
 
    return parser.parse_args()
    
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    
    # Calculate mean/std with data_dir from args
    mean, std = calculate_mean_std(data_dir=args.data_dir) 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    train_dataset = SUN397Dataset(data_dir=args.data_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    
    model = CNN(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                      factor=0.5, patience=3)
    
    best_acc = 0.0
    epochs_no_improve = 0
    patience = 7
    
    print("Starting training...\n")
    for epoch in range(1, 31):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, epoch)
        print(f"Epoch {epoch}/30 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        
        scheduler.step(train_acc)
        
        if train_acc > best_acc:
            best_acc = train_acc
            torch.save(model.state_dict(), 'model.pt')
            print(f"  ✓ Best model saved! Accuracy: {best_acc:.2f}%")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {patience} epochs)")
            break
        
        current_lr = optimizer.param_groups[0]['lr']
        if epoch % 5 == 0:
            print(f"  Current learning rate: {current_lr:.6f}")
    
    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"Best accuracy: {best_acc:.2f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
