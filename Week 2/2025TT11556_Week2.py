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

# THE_BLACK_PEARL_12

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandAugment(),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading dataset...")
# Load the full training dataset
# The unzipped folder 'train' contains a subfolder 'train' which has the class folders
full_trainset = torchvision.datasets.ImageFolder(root='./train', transform=train_transform)

# Split into train and validation (80/20)
train_size = int(0.8 * len(full_trainset))
val_size = len(full_trainset) - train_size

# Set seed for reproducibility
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(full_trainset, [train_size, val_size], generator=generator)

full_valset = torchvision.datasets.ImageFolder(root='./train', transform=test_transform)
val_dataset.dataset = full_valset

trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
valloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# test dataset stuff
class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        # We need to list all files in the test directory
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

testset = TestDataset(img_dir='./test_set', transform=test_transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)


model = torchvision.models.resnet18(num_classes=10)
model = model.to(device)


criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35)

epochs = 35
min_val_loss = float('inf')
best_weights = copy.deepcopy(model.state_dict())

print("\nStarting Training...")
for epoch in range(epochs):
    # --- Training Phase ---
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
        
    t_loss = running_loss / len(train_dataset)
    t_acc = correct / total
    
    # --- Validation Phase ---
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
            
    v_loss = val_loss / len(val_dataset)
    v_acc = val_correct / val_total
    
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}")
    
    # Step the scheduler
    scheduler.step()
    
    # save if better
    if v_loss < min_val_loss:
        print(f"--> Val loss improved from {min_val_loss:.4f} to {v_loss:.4f}. Saving model!")
        min_val_loss = v_loss
        best_weights = copy.deepcopy(model.state_dict())

# Load the best weights back into the model before testing
model.load_state_dict(best_weights)
torch.save(model.state_dict(), "best_resnet18.pth")




print("\nGenerating Predictions on Test Set...")
model.eval()
predictions = []

with torch.no_grad():
    for images, img_ids in testloader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        for img_id, pred in zip(img_ids, predicted):
            predictions.append({
                "img_id": img_id,
                "label": pred.item()
            })

csv_filename = "2025TT11556_Week2_CNN.csv"
df = pd.DataFrame(predictions)
df.to_csv(csv_filename, index=False)
print(f"Saved submission to {csv_filename}!")
