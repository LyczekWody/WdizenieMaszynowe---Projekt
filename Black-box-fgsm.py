import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
from PIL import Image

# Urządzenie
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformacja
transform = transforms.Compose([
    transforms.ToTensor()
])

# Dane
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# ====== MODEL GŁÓWNY ======
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.network(x)

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trening głównego modelu
for epoch in range(5):
    model.train()
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"[MODEL] Epoch {epoch+1}/5 complete")

# ====== MODEL ZASTĘPCZY ======
class AlternativeCNN(nn.Module):
    def __init__(self):
        super(AlternativeCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

substitute_model = AlternativeCNN().to(device)
sub_optimizer = optim.Adam(substitute_model.parameters(), lr=0.001)

# Trening modelu zastępczego
for epoch in range(5):
    substitute_model.train()
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        sub_optimizer.zero_grad()
        outputs = substitute_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        sub_optimizer.step()
    print(f"[SUB] Epoch {epoch+1}/5 complete")

# ====== FUNKCJA FGSM ======
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)

# ====== TEST: Atak na SUB, ocena na MODEL (BLACK-BOX) ======
def blackbox_attack(testloader, epsilon):
    correct = 0
    adv_examples = []

    substitute_model.eval()
    model.eval()

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True

        outputs = substitute_model(images)
        loss = criterion(outputs, labels)
        substitute_model.zero_grad()
        loss.backward()
        data_grad = images.grad.data

        # Atak FGSM
        perturbed_data = fgsm_attack(images, epsilon, data_grad)

        # Ocena na modelu głównym (black-box)
        outputs = model(perturbed_data)
        _, final_pred = outputs.max(1)

        correct += (final_pred == labels).sum().item()

        # Przykład do pokazania
        if len(adv_examples) < 5:
            adv_examples.append((perturbed_data[0].detach().cpu(), final_pred[0].item(), labels[0].item()))

    final_acc = correct / len(testloader.dataset)
    print(f"Epsilon: {epsilon}\tBlack-box Accuracy = {final_acc * 100:.2f}%")

    # Pokaż przykłady
    plt.figure(figsize=(10, 2))
    for i, (img, pred, label) in enumerate(adv_examples):
        plt.subplot(1, 5, i+1)
        plt.imshow(img.squeeze(), cmap="gray")
        plt.title(f"P:{pred} / T:{label}")
        plt.axis('off')
    plt.suptitle(f"Przykładowe adwersarialne obrazy (ε={epsilon})")
    plt.tight_layout()
    plt.show()

# ====== URUCHOMIENIE BLACK-BOX ATAKU ======
for eps in [0.05, 0.1, 0.2, 0.5,]:
    blackbox_attack(testloader, epsilon=eps)
