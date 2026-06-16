import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
from PIL import Image
import os
import copy
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# SMALL_WEEK_3

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# ==========================================
# 1. DATA PIPELINE SETUP
# ==========================================

# Base transform for validation/testing
base_transform = transforms.Compose([
    transforms.Resize((224, 224)), # Use 224x224 because pretrained ImageNet models expect it
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Build 1 (Frozen Backbone) likes lighter augmentation
light_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Build 2 (Fine-Tuning) can handle heavier augmentation
heavy_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandAugment(),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading dataset...")
# Load the full dataset (we will manipulate transforms in the loop)
full_trainset = torchvision.datasets.ImageFolder(root='./train')

# Split into train and validation (80/20)
train_size = int(0.8 * len(full_trainset))
val_size = len(full_trainset) - train_size
generator = torch.Generator().manual_seed(42)

# Create validation subset directly
val_dataset = copy.deepcopy(full_trainset)
val_dataset.transform = base_transform
# Actually, Subset doesn't allow direct transform changing easily without a wrapper, so we load twice:

train_subset_light, _ = random_split(torchvision.datasets.ImageFolder(root='./train', transform=light_transform), [train_size, val_size], generator=generator)
train_subset_heavy, _ = random_split(torchvision.datasets.ImageFolder(root='./train', transform=heavy_transform), [train_size, val_size], generator=generator)
_, val_subset = random_split(torchvision.datasets.ImageFolder(root='./train', transform=base_transform), [train_size, val_size], generator=generator)

# DataLoaders
valloader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=0)

class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB') 
        if self.transform:
            image = self.transform(image)
        return image, img_name

testset = TestDataset(img_dir='./test_set', transform=base_transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)


# ==========================================
# 2. TRAINING FUNCTION
# ==========================================

def run_training(model_name, model, trainloader, lr, epochs):
    print(f"\n--- Starting {model_name} ---")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # We only pass parameters that require gradients to the optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        t_loss = running_loss / train_size
        t_acc = correct / total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        v_loss = val_loss / val_size
        v_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs} | T_Loss: {t_loss:.4f} T_Acc: {t_acc:.4f} | V_Loss: {v_loss:.4f} V_Acc: {v_acc:.4f}")
        
        scheduler.step()
        
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_weights = copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_weights)
    return model

def generate_predictions(model, csv_filename):
    print(f"Generating Predictions for {csv_filename}...")
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, img_ids in testloader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for img_id, pred in zip(img_ids, predicted):
                predictions.append({"img_id": img_id, "label": pred.item()})
                
    df = pd.DataFrame(predictions)
    df.to_csv(csv_filename, index=False)
    print(f"Saved {csv_filename}!\n")

# ==========================================
# 3. BUILD 1: FROZEN BACKBONE
# ==========================================

model1 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
# Freeze the backbone
for param in model1.parameters():
    param.requires_grad = False
# Replace the classification head (this layer is unfrozen by default)
model1.fc = nn.Linear(model1.fc.in_features, 10)
model1 = model1.to(device)

trainloader_light = DataLoader(train_subset_light, batch_size=64, shuffle=True, num_workers=0)

# Train Build 1 (Since we only train 1 layer, we can do 100 epochs very fast)
model1 = run_training("BUILD 1 (Frozen Backbone)", model1, trainloader_light, lr=0.005, epochs=100)
generate_predictions(model1, "2025TT11556_Build1_Frozen.csv")


# ==========================================
# 4. BUILD 2: FULL FINE-TUNING
# ==========================================

model2 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
# Do NOT freeze the backbone
model2.fc = nn.Linear(model2.fc.in_features, 10)
model2 = model2.to(device)

trainloader_heavy = DataLoader(train_subset_heavy, batch_size=64, shuffle=True, num_workers=0)

# Train Build 2 (Train everything with a very low LR to protect pretrained features)
model2 = run_training("BUILD 2 (Full Fine-Tuning)", model2, trainloader_heavy, lr=1e-4, epochs=50)
generate_predictions(model2, "2025TT11556_Build2_FineTuned.csv")

print("All tasks successfully completed!")
