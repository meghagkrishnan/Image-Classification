import pytest
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from src.model import VGG16  # Assuming your model is in 'model.py'

# Define a test directory (this should point to a valid folder with images for testing)
data_dir = "/home/gopalakrishnanm/Tutorial_Projects/Image-Classification/data/PetImages"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data transformation and dataset loading
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 (VGG input size)
    transforms.ToTensor(),  # Convert images to tensor
])

# Create the dataset and split it
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
# Calculate half the size of the dataset
half_size = len(full_dataset) // 2

# Create a subset with the first half of the dataset
indices = list(range(half_size))
dataset = Subset(full_dataset, indices)

train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # 20% for testing
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Test 1: Verify if dataset splits correctly
def test_data_loading():
    assert len(train_dataset) == int(0.8 * len(dataset)), f"Expected {int(0.8 * len(dataset))}, but got {len(train_dataset)}"
    assert len(test_dataset) == len(dataset) - int(0.8 * len(dataset)), f"Expected {len(dataset) - int(0.8 * len(dataset))}, but got {len(test_dataset)}"

# Test 2: Check if model instantiation works correctly
def test_model_instantiation():
    model = VGG16(num_classes=2).to(device)
    # Check if model is not None
    assert model is not None, "Model is None, expected an instance of the model"

# Test 3: Check if forward pass works
def test_forward_pass():
    model = VGG16(num_classes=2).to(device)
    model.eval()  # Switch model to evaluation mode
    dummy_input = torch.randn(1, 3, 224, 224).to(device)  # Dummy input tensor (batch size = 1, 3 channels, 224x224 image)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (1, 2), f"Expected output shape (1, 2), but got {output.shape}"

# Test 4: Check if the training loop runs without errors (does not check performance)
def test_training_loop():
    model = VGG16(num_classes=2).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

    for epoch in range(1):  # Run for 1 epoch
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i == 5:  # Run only a few steps for the test
                break
        break  # End the epoch after the first iteration to keep the test fast

    assert loss.item() > 0, "Loss is zero, which might indicate an issue in the forward pass"

# Test 5: Evaluate the model's accuracy on the test set
def test_evaluation():
    model = VGG16(num_classes=2).to(device)
    model.eval()  # Set model to evaluation mode

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    assert accuracy >= 0, f"Accuracy should be non-negative, but got {accuracy}%"
    assert accuracy <= 100, f"Accuracy should be <= 100%, but got {accuracy}%"
