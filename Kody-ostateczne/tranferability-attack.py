import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Urządzenie (GPU lub CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformacja danych (zamiana obrazów na tensory)
transform = transforms.ToTensor()

# Ładowanie zbioru MNIST
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=1000, shuffle=False)

# Definicja modelu SimpleCNN - prosty konwolucyjny sieć neuronowa
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

# Definicja modelu LeNet - starsza, klasyczna architektura CNN
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

# Funkcja treningowa modelu
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

# Implementacja ataku FGSM (Fast Gradient Sign Method)
def fgsm_attack(model, images, labels, epsilon):
    images.requires_grad = True
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()
    grad_sign = images.grad.data.sign()
    adv_images = torch.clamp(images + epsilon * grad_sign, 0, 1)  # Tworzenie przykładów adwersarialnych
    return adv_images.detach()

# Implementacja ataku PGD (Projected Gradient Descent)
def pgd_attack(model, images, labels, epsilon, alpha=0.01, iters=20):
    original_images = images.clone().detach()
    adv_images = images.clone().detach()

    for _ in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()
        adv_images = adv_images + alpha * adv_images.grad.sign()
        # Ograniczenie perturbacji do epsilona
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(original_images + eta, 0, 1).detach()

    return adv_images

# Funkcja przeprowadzająca atak i oceniająca skuteczność modelu ofiary
# Funkcja przeprowadzająca atak i oceniająca skuteczność modelu ofiary
from skimage.metrics import structural_similarity as ssim

def attack(loader, model_attacker, model_victim, epsilon, show_examples=False):
    model_attacker.eval()
    model_victim.eval()

    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    adv_fgsm = fgsm_attack(model_attacker, images, labels, epsilon)
    adv_pgd  = pgd_attack(model_attacker, images, labels, epsilon)

    # Obliczanie dokładności na modelu ofiary
    with torch.no_grad():
        out_fgsm = model_victim(adv_fgsm)
        out_pgd  = model_victim(adv_pgd)
        acc_fgsm = (out_fgsm.argmax(1) == labels).float().mean().item() * 100
        acc_pgd  = (out_pgd.argmax(1) == labels).float().mean().item() * 100

    # Obliczanie SSIM dla pierwszych N przykładów (np. 100)
    ssim_scores_fgsm = []
    ssim_scores_pgd  = []
    N = min(100, images.size(0))
    orig = images.detach().cpu().numpy()
    advf = adv_fgsm.detach().cpu().numpy()
    advp = adv_pgd.detach().cpu().numpy()
    for i in range(N):
        orig_img = orig[i].squeeze()
        advf_img = advf[i].squeeze()
        advp_img = advp[i].squeeze()
        ssim_scores_fgsm.append(ssim(orig_img, advf_img, data_range=1.0))
        ssim_scores_pgd.append(ssim(orig_img, advp_img, data_range=1.0))

    mean_ssim_fgsm = np.mean(ssim_scores_fgsm)
    mean_ssim_pgd  = np.mean(ssim_scores_pgd)

    print(f"Epsilon={epsilon:.2f} | FGSM acc = {acc_fgsm:.2f}%, SSIM = {mean_ssim_fgsm:.3f} | "
          f"PGD acc = {acc_pgd:.2f}%, SSIM = {mean_ssim_pgd:.3f}")

    if show_examples:
        plt.figure(figsize=(10, 4))
        for i in range(6):
            plt.subplot(2, 6, i + 1)
            plt.imshow(adv_fgsm[i].cpu().squeeze(), cmap="gray")
            plt.title(f"FGSM:{out_fgsm[i].argmax().item()}")
            plt.axis("off")

            plt.subplot(2, 6, i + 7)
            plt.imshow(adv_pgd[i].cpu().squeeze(), cmap="gray")
            plt.title(f"PGD:{out_pgd[i].argmax().item()}")
            plt.axis("off")
        plt.suptitle(f"Adversarial examples (ε={epsilon})")
        plt.tight_layout()
        plt.show()



# Tworzenie i trenowanie modeli
model_attacker = SimpleCNN().to(device)
model_victim = LeNet().to(device)

print("Trenuję model atakujący (SimpleCNN)...")
train_model(model_attacker, trainloader)

print("Trenuję model ofiarę (LeNet)...")
train_model(model_victim, trainloader)

# Funkcja do oceny skuteczności modelu na danych testowych
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

print("Skuteczność LeNet na czystych danych:", evaluate(model_victim, testloader), "%")

# Testy ataków dla różnych wartości epsilon
epsilons = [0.05, 0.1, 0.2, 0.5]
for eps in epsilons:
    attack(testloader, model_attacker=model_attacker, model_victim=model_victim, epsilon=eps, show_examples=True)

