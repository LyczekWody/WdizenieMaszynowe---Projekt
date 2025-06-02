import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np


# ==== Przygotowanie ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=1000, shuffle=False)


# ==== MODELE ====
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, 5), nn.Tanh(), nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(256, 120), nn.Tanh(),
            nn.Linear(120, 84), nn.Tanh(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.net(x)


# ==== Funkcja treningowa ====
def train_model(model, loader, epochs=3):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


# ==== Ataki ====
def fgsm_attack(model, images, labels, epsilon):
    images.requires_grad = True
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()
    grad_sign = images.grad.data.sign()
    adv_images = torch.clamp(images + epsilon * grad_sign, 0, 1)
    return adv_images.detach()

def pgd_attack(model, images, labels, epsilon, alpha=0.01, iters=10):
    original_images = images.clone().detach()
    adv_images = images.clone().detach()

    for _ in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()
        adv_images = adv_images + alpha * adv_images.grad.sign()
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(original_images + eta, 0, 1).detach()

    return adv_images


# ==== Ewaluacja ====
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


# ==== Trening modeli ====
model_attacker = SimpleCNN().to(device)
model_victim = LeNet().to(device)

print("Trenuję model atakujący (SimpleCNN)...")
train_model(model_attacker, trainloader)

print("Trenuję model ofiarę (LeNet)...")
train_model(model_victim, trainloader)


# ==== Ataki ====
model_attacker.eval()

images, labels = next(iter(testloader))
images, labels = images.to(device), labels.to(device)

print("Generuję przykłady FGSM...")
adv_fgsm = fgsm_attack(model_attacker, images, labels, epsilon=0.3)

print("Generuję przykłady PGD...")
adv_pgd = pgd_attack(model_attacker, images, labels, epsilon=0.3, alpha=0.01, iters=20)


# ==== Testowanie na ofierze ====
print("Test na czystych danych (LeNet):", evaluate(model_victim, testloader), "%")

with torch.no_grad():
    outputs_fgsm = model_victim(adv_fgsm)
    acc_fgsm = (outputs_fgsm.argmax(1) == labels).sum().item() / labels.size(0) * 100

    outputs_pgd = model_victim(adv_pgd)
    acc_pgd = (outputs_pgd.argmax(1) == labels).sum().item() / labels.size(0) * 100

print(f"Skuteczność LeNet na FGSM (transfer): {acc_fgsm:.2f}%")
print(f"Skuteczność LeNet na PGD  (transfer): {acc_pgd:.2f}%")
