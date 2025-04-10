import torch
import torchvision
import torchvision.transforms as transforms

from resnet import make_resnet18k


# Hyperparameters
epochs = 4000
lr = 0.0001
p = 0.1  # can also try 0 or 0.2
batch_size = 128  # all experiments use this
k = 1  # width parameter for ResNet18

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Prepare CIFAR10 dataset
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# Add label noise
num_datapoints = len(trainset)
noisy_indices = torch.randperm(num_datapoints)[:int(p * num_datapoints)]  # indices of noisy labels

for idx in noisy_indices:
    new_label = torch.randint(0, len(classes), size=(1,)).item()
    while new_label == trainset.targets[idx]:
        new_label = torch.randint(0, len(classes), size=(1,)).item()
    trainset.targets[idx] = new_label

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=512,
                                         shuffle=False)

# Prepare ResNet model
model = make_resnet18k(k=1, num_classes=len(classes))  # k=1 for 1x ResNet18

# Check if GPU is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(trainloader):.4f}")

correct_predictions = 0
for i, data in enumerate(testloader):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    predicted_labels = torch.argmax(outputs, dim=1)

    correct_predictions += (predicted_labels == labels).sum().item()

print(f"Accuracy of the model on the test set: {100 * correct_predictions / len(testset):.2f}%")




