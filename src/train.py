import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

from model import VGG16

data_dir = "/home/gopalakrishnanm/Tutorial_Projects/Image-Classification/data/PetImages"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 (VGG input size)
    transforms.ToTensor(),  # Convert images to tensor
])

full_dataset = datasets.ImageFolder(root = data_dir, transform = transform)

# Split dataset into 80% training and 20% testing
train_size = int(0.8 * len(full_dataset))  # 80% for training
test_size = len(full_dataset) - train_size  # 20% for testing
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of testing samples: {len(test_dataset)}")

#model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
model = VGG16(num_classes = 2).to(device)

# Loss and Optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.005)

print(len(train_loader))

epochs = 1
for epoch in range(epochs):
    for i, (images,labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 5 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                .format(epoch+1, epochs, i+1, len(train_loader), loss.item()))
# images, labels = next(iter(data_loader))
# print(images.shape)
# images = images.numpy().transpose((0, 2, 3, 1))
# # plt.imshow(images[0])
# # plt.show()
# print(images.shape, labels.shape)
# print(labels[0])

# Validation
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        #del images, labels, outputs

    print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))