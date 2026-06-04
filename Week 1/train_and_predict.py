"""
# Week 1 Assignment: My First PyTorch Models

This week was mostly about getting used to PyTorch syntax and building basic networks. I started with a normal fully connected network (MLP) and then moved on to a CNN. Honestly, the hardest part was just keeping track of tensor shapes, especially when figuring out the input size for the linear layer after the convolutions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import os

# checking for gpu. it took forever when I accidentally ran it on cpu earlier lol
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

"""
### Loading the Data
I went with a batch size of 64. I tried 32 at first but it was kind of slow, and 128 didn't seem to make much difference in speed. Normalizing to 0.5 mean and 0.5 std is what the PyTorch tutorial suggested for grayscale stuff, so I just stuck with that.
"""

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("Downloading FashionMNIST...")
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# custom dataset for the test set since it's just a bunch of loose images
class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        # grabbing exactly 100 images based on the sample submission
        self.img_names = sorted([f"{i:03d}.png" for i in range(100)])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # got a weird bug earlier where it read as 3 channels. forcing it to grayscale fixes it.
        image = Image.open(img_path).convert('L') 
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_name[:3] 

testset = TestDataset(img_dir='./images', transform=transform)
# test batch size doesn't matter much so 10 is fine
testloader = DataLoader(testset, batch_size=10, shuffle=False)

"""
### Build 1: Fully Connected Network (MLP)
For the first build, I just flattened the 28x28 image into 784 pixels. I added a couple of hidden layers (512 and 256). Used ReLU because that seems to be the default activation everyone uses. It works pretty well.
"""

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

"""
### Build 2: Convolutional Neural Network (CNN)
For the CNN, I used two Conv2d layers. Finding the linear layer input size was annoying. 
Image starts at 28x28.
Conv1 (padding=1) keeps it 28x28. MaxPool halves it to 14x14.
Conv2 (padding=1) keeps it 14x14. MaxPool halves it to 7x7.
Since I have 64 channels in the second conv layer, the flattened size is 64 * 7 * 7.
"""

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 1 input channel (grayscale), 32 output channels, 3x3 filter
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
        self.flatten = nn.Flatten()
        # 64 channels * 7x7 spatial size after two pools
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

"""
### Training Loop
I wrote a helper function so I don't have to copy-paste the training loop twice. I chose a learning rate of 0.001 with Adam. I tried 0.01 at first but the loss was bouncing around too much. 5 epochs seems like enough for FashionMNIST to get decent results without waiting all day.
"""

def train_model(model, epochs, name):
    print(f"\nTraining {name}...")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # don't forget to zero the gradients! forgot this once and the loss exploded
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 300 == 299:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}], Loss: {running_loss / 300:.3f}')
                running_loss = 0.0
    print(f"Finished Training {name}")
    return model

"""
### Inference and Submissions
This part just runs the test images through the models and saves them to CSV.
"""

def generate_submission(model, filename):
    model.eval() # super important to turn off dropout/batchnorm for testing
    predictions = []
    
    with torch.no_grad():
        for images, img_ids in testloader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            for img_id, pred in zip(img_ids, predicted):
                predictions.append({"image_id": img_id, "label": pred.item()})
                
    df = pd.DataFrame(predictions)
    df.to_csv(filename, index=False)
    print(f"Saved {filename}")

epochs = 5

# train and save MLP
mlp_model = train_model(MLP(), epochs, "MLP")
generate_submission(mlp_model, "2025TT11556_NN.csv")

# train and save CNN
cnn_model = train_model(CNN(), epochs, "CNN")
generate_submission(cnn_model, "2025TT11556_CNN.csv")

"""
### Final Thoughts
The CNN performed better than the fully connected network, which was expected because it actually looks at spatial patterns instead of just a flattened list of pixels. The loss dropped much faster on the CNN. The MLP is okay, but you can tell it's hitting a wall compared to the convolutions.
"""
