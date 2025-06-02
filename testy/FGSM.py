import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
from PIL import Image
import time
from skimage.metrics import structural_similarity as ssim
import numpy as np


# Konfiguracja urządzenia
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformacja
transform = transforms.Compose([
    transforms.ToTensor()
])

# Ładowanie danych
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Prosty model sieci
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

# Inicjalizacja modelu, loss, optymalizatora
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trening modelu
for epoch in range(6):
    model.train()
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/6, Loss: {running_loss:.4f}")

# Testowanie modelu
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Accuracy on test set: {100 * correct / total:.2f}%")

# Definicja ataku FGSM
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# Funkcja ataku na cały testloader
#def attack(testloader, epsilon=0.1, show_examples=False):
    correct = 0
    total = 0
    examples = []
    ssim_scores = []
    model.eval()

    start_time = time.time()

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True

        outputs = model(images)
        init_preds = outputs.argmax(1)

        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()

        data_grad = images.grad.data
        perturbed_data = fgsm_attack(images, epsilon, data_grad)

        with torch.no_grad():
            output = model(perturbed_data)
            final_preds = output.argmax(1)

        correct += (final_preds == labels).sum().item()
        total += labels.size(0)

        # Liczenie SSIM (dla pierwszych 100 obrazów)
        for i in range(min(100 - len(ssim_scores), images.size(0))):
            orig = images[i].squeeze().detach().cpu().numpy()
            pert = perturbed_data[i].squeeze().detach().cpu().numpy()
            s = ssim(orig, pert, data_range=1.0)
            ssim_scores.append(s)

        # Przykłady do wizualizacji
        if show_examples and len(examples) < 5:
            for i in range(min(5 - len(examples), images.size(0))):
                orig = images[i].squeeze().detach().cpu().numpy()
                pert = perturbed_data[i].squeeze().detach().cpu().numpy()
                examples.append((orig, init_preds[i].item(), pert, final_preds[i].item(), labels[i].item()))

        if len(ssim_scores) >= 100:
            break  # Wystarczy 100 SSIM

    end_time = time.time()
    duration = end_time - start_time
    accuracy = 100. * correct / total
    mean_ssim = np.mean(ssim_scores)

    print(f"Epsilon: {epsilon:.2f} - Accuracy after FGSM attack: {accuracy:.2f}%")
    print(f"  ⏱ Czas działania: {duration:.2f} s")
    print(f"  🔎 Średni SSIM (dla 100 próbek): {mean_ssim:.4f}")

    # Wizualizacja
    if show_examples:
        plt.figure(figsize=(10, 4))
        for idx, (orig, init_pred, pert, final_pred, label) in enumerate(examples):
            plt.subplot(2, 5, idx + 1)
            plt.imshow(orig, cmap='gray')
            plt.title(f'P: {init_pred}')
            plt.axis('off')

            plt.subplot(2, 5, idx + 6)
            plt.imshow(pert, cmap='gray')
            plt.title(f'P: {final_pred}')
            plt.axis('off')
        plt.suptitle(f"Przewidywania przed (góra) i po (dół) ataku FGSM (ε={epsilon})")
        plt.tight_layout()
        plt.show()
    correct = 0
    total = 0
    examples = []
    model.eval()

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True

        outputs = model(images)
        init_preds = outputs.argmax(1)

        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()

        data_grad = images.grad.data
        perturbed_data = fgsm_attack(images, epsilon, data_grad)

        with torch.no_grad():
            output = model(perturbed_data)
            final_preds = output.argmax(1)

        correct += (final_preds == labels).sum().item()
        total += labels.size(0)

        if show_examples and len(examples) < 5:
            for i in range(min(5 - len(examples), images.size(0))):
                orig = images[i].squeeze().detach().cpu().numpy()
                pert = perturbed_data[i].squeeze().detach().cpu().numpy()
                examples.append((orig, init_preds[i].item(), pert, final_preds[i].item(), labels[i].item()))

    accuracy = 100. * correct / total
    print(f"Epsilon: {epsilon:.2f} - Accuracy after FGSM attack: {accuracy:.2f}%")

    if show_examples:
        plt.figure(figsize=(10, 4))
        for idx, (orig, init_pred, pert, final_pred, label) in enumerate(examples):
            plt.subplot(2, 5, idx + 1)
            plt.imshow(orig, cmap='gray')
            plt.title(f'P: {init_pred}')
            plt.axis('off')

            plt.subplot(2, 5, idx + 6)
            plt.imshow(pert, cmap='gray')
            plt.title(f'P: {final_pred}')
            plt.axis('off')
        plt.suptitle(f"Przewidywania przed (góra) i po (dół) ataku FGSM (ε={epsilon})")
        plt.tight_layout()
        plt.show()
# Funkcja ataku na cały testloader
def attack(testloader, epsilon=0.1, show_examples=False):
    correct = 0
    total = 0
    examples = []
    ssim_scores = []
    model.eval()

    start_time = time.time()

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True

        outputs = model(images)
        init_preds = outputs.argmax(1)

        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()

        data_grad = images.grad.data
        perturbed_data = fgsm_attack(images, epsilon, data_grad)

        with torch.no_grad():
            output = model(perturbed_data)
            final_preds = output.argmax(1)

        correct += (final_preds == labels).sum().item()
        total += labels.size(0)

        # Liczenie SSIM (dla pierwszych 100 obrazów)
        for i in range(min(100 - len(ssim_scores), images.size(0))):
            orig = images[i].squeeze().detach().cpu().numpy()
            pert = perturbed_data[i].squeeze().detach().cpu().numpy()
            s = ssim(orig, pert, data_range=1.0)
            ssim_scores.append(s)

        # Przykłady do wizualizacji
        if show_examples and len(examples) < 5:
            for i in range(min(5 - len(examples), images.size(0))):
                orig = images[i].squeeze().detach().cpu().numpy()
                pert = perturbed_data[i].squeeze().detach().cpu().numpy()
                examples.append((orig, init_preds[i].item(), pert, final_preds[i].item(), labels[i].item()))

        if len(ssim_scores) >= 100:
            break  # Wystarczy 100 SSIM

    end_time = time.time()
    accuracy = 100. * correct / total
    mean_ssim = np.mean(ssim_scores)

    print(f"Epsilon: {epsilon:.2f} - Accuracy after FGSM attack: {accuracy:.2f}%")
    print(f"  ⏱ Czas działania: {end_time - start_time:.2f} s")
    print(f"  🔎 Średni SSIM (dla 100 próbek): {mean_ssim:.4f}")

    if show_examples:
        plt.figure(figsize=(10, 4))
        for idx, (orig, init_pred, pert, final_pred, label) in enumerate(examples):
            plt.subplot(2, 5, idx + 1)
            plt.imshow(orig, cmap='gray')
            plt.title(f'P: {init_pred}')
            plt.axis('off')

            plt.subplot(2, 5, idx + 6)
            plt.imshow(pert, cmap='gray')
            plt.title(f'P: {final_pred}')
            plt.axis('off')
        plt.suptitle(f"Przewidywania przed (góra) i po (dół) ataku FGSM (ε={epsilon})")
        plt.tight_layout()
        plt.show()

# Uruchomienie testów dla różnych epsilonów
epsilons = [0.05, 0.1, 0.2, 0.5]
for eps in epsilons:
    attack(testloader, epsilon=eps, show_examples=(eps == 0.2))
