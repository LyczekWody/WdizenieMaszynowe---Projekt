import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
from PIL import Image

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
for epoch in range(13):
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
    print(f"Epoch {epoch+1}/5, Loss: {running_loss:.4f}")

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

# Wyświetlenie 5 losowych obrazów z predykcjami przed i po ataku
examples = list(zip(testset.data, testset.targets))  # dane + etykiety
model.eval()

epsilon = 0.2  # Możesz dostosować wartość epsilon

perturbed_correct = 0  # Liczba poprawnych klasyfikacji po ataku
perturbed_total = 0  # Łączna liczba próbek po ataku

for i in range(5):  # Wyświetlamy 5 obrazków
    idx = random.randint(0, len(examples) - 1)
    image, label = examples[idx]
    pil_image = Image.fromarray(image.numpy(), mode='L')
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Predykcja przed atakiem
    with torch.no_grad():
        output = model(input_tensor)
        init_pred = output.argmax(1).item()

    # Zabezpieczenie przed obliczaniem gradientu dla obrazu
    input_tensor.requires_grad = True

    # Obliczanie straty i gradientu
    output = model(input_tensor)
    loss = criterion(output, torch.tensor([label]).to(device))
    model.zero_grad()
    loss.backward()

    # Atak FGSM
    data_grad = input_tensor.grad.data
    perturbed_image = fgsm_attack(input_tensor, epsilon, data_grad)

    # Predykcja po ataku
    with torch.no_grad():
        output = model(perturbed_image)
        final_pred = output.argmax(1).item()

    # Aktualizacja liczby poprawnych predykcji po ataku
    perturbed_total += 1
    if final_pred == label:
        perturbed_correct += 1

    # Wyświetlanie wyników
    plt.subplot(2, 5, i+1)
    plt.imshow(image, cmap='gray')
    plt.title(f'P: {init_pred}')
    plt.axis('off')

    plt.subplot(2, 5, i+6)
    perturbed_image = perturbed_image.squeeze().cpu().detach().numpy()
    plt.imshow(perturbed_image, cmap='gray')
    plt.title(f'P: {final_pred}')
    plt.axis('off')

plt.suptitle("Przewidywania przed (góra) i po (dół) ataku FGSM")
plt.tight_layout()
plt.show()

# Obliczanie dokładności po ataku
perturbed_accuracy = 100 * perturbed_correct / perturbed_total
print(f"Accuracy after FGSM attack: {perturbed_accuracy:.2f}%")
