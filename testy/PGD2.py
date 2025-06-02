import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchattacks import PGD
import matplotlib.pyplot as plt
import torch.optim as optim
import time
from skimage.metrics import structural_similarity as ssim



# 1. Parametry i urządzenie
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 100

# 2. Wczytaj dane MNIST
transform = transforms.Compose([
    transforms.ToTensor()
])

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# 3. Prosty model CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(5408, 10)
        )

    def forward(self, x):
        return self.net(x)

# 4. Załaduj wytrenowany model lub wytrenuj nowy
model = SimpleCNN().to(device)
# Wczytaj model, jeśli masz zapisany plik
# model.load_state_dict(torch.load('mnist_cnn.pth'))


#4.5 

# Parametry treningu
num_epochs = 5
learning_rate = 0.001



# Funkcja trenowania modelu
def train(model, train_loader, num_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    return train_losses, train_accuracies

model.eval()

# Trenowanie modelu
train_losses, train_accuracies = train(model, train_loader, num_epochs, learning_rate)

# 5. Funkcja do oceny skuteczności
def evaluate(model, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Accuracy: {acc:.2f}%")
    return acc

# 6. Ocena na czystych danych
print(">> Evaluating on clean data")
clean_acc = evaluate(model, test_loader)

# 7. Tworzenie ataku PGD
attack = PGD(model, eps=0.2, alpha=2/255, steps=40, random_start=True)

# 8. Ewaluacja modelu pod atakiem
def evaluate_under_attack(model, loader, attack):
    correct = 0
    total = 0
    measured = False
    total_ssim = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if not measured:
            start_time = time.time()
            adv_images = attack(images, labels)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f">> Time to generate adversarial examples for 100 images: {elapsed_time:.4f} seconds")

            # Oblicz SSIM dla każdego obrazu (tylko dla pierwszego batcha)
            ssim_scores = []
            for i in range(images.size(0)):
                img_clean = images[i].squeeze().detach().cpu().numpy()
                img_adv = adv_images[i].squeeze().detach().cpu().numpy()
                score = ssim(img_clean, img_adv, data_range=1.0)
                ssim_scores.append(score)
            avg_ssim = sum(ssim_scores) / len(ssim_scores)
            print(f">> Average SSIM (clean vs adversarial) for first 100 images: {avg_ssim:.4f}")
            measured = True
        else:
            adv_images = attack(images, labels)

        outputs = model(adv_images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    acc = 100 * correct / total
    print(f"Adversarial Accuracy (PGD): {acc:.2f}%")
    return acc


print(">> Evaluating under PGD attack")
adv_acc = evaluate_under_attack(model, test_loader, attack)

# 9. Wizualizacja kilku przykładów przed i po ataku PGD
def visualize_attack(model, loader, attack, num_images=5):
    data_iter = iter(loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    # Generujemy obrazy adversarialne
    adv_images = attack(images, labels)
    
    # Przekształcamy do numpy
    images_np = images[:num_images].cpu().numpy()
    adv_images_np = adv_images[:num_images].detach().cpu().numpy()
    labels_np = labels[:num_images].cpu().numpy()

    # Wyświetlenie
    fig, axes = plt.subplots(num_images, 2, figsize=(5, 2 * num_images))
    fig.suptitle("Clean vs Adversarial Images (PGD)", fontsize=16)
    for i in range(num_images):
        # Oryginalny obraz
        axes[i, 0].imshow(images_np[i][0], cmap='gray')
        axes[i, 0].set_title(f"Clean - Label: {labels_np[i]}")
        axes[i, 0].axis('off')

        # Adversarialny obraz
        axes[i, 1].imshow(adv_images_np[i][0], cmap='gray')
        with torch.no_grad():
            pred = model(adv_images[i].unsqueeze(0)).argmax().item()
        axes[i, 1].set_title(f"Adversarial - Pred: {pred}")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# 10. Porównanie wyników dla różnych epsilonów
epsilons = [0.05, 0.1, 0.2, 0.5]
results = []

print("\n>> Evaluating model under PGD attack with different epsilons:")

def visualize_epsilon_comparison(images, adv_images, labels, epsilon, model, num_images=5):
    images_np = images[:num_images].cpu().numpy()
    adv_images_np = adv_images[:num_images].detach().cpu().numpy()
    labels_np = labels[:num_images].cpu().numpy()

    fig, axes = plt.subplots(2, num_images, figsize=(2 * num_images, 4))
    fig.suptitle(f"Clean vs Adversarial Images (ε = {epsilon})", fontsize=16)

    for i in range(num_images):
        # Clean image
        axes[0, i].imshow(images_np[i][0], cmap='gray')
        axes[0, i].set_title(f"Label: {labels_np[i]}")
        axes[0, i].axis('off')

        # Adversarial image
        axes[1, i].imshow(adv_images_np[i][0], cmap='gray')
        with torch.no_grad():
            pred = model(adv_images[i].unsqueeze(0)).argmax().item()
        axes[1, i].set_title(f"Adv Pred: {pred}")
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel("Clean", fontsize=12)
    axes[1, 0].set_ylabel("Adversarial", fontsize=12)

    plt.tight_layout()
    plt.show()


for eps in epsilons:
    print(f"\n=== Epsilon: {eps} ===")
    attack = PGD(model, eps=eps, alpha=2/255, steps=40, random_start=True)

    # Pomiar accuracy i SSIM
    correct = 0
    total = 0
    ssim_scores = []

    # Jeden batch do SSIM
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    start_time = time.time()
    adv_images = attack(images, labels)
    visualize_epsilon_comparison(images, adv_images, labels, epsilon=eps, model=model)
    end_time = time.time()

    for i in range(images.size(0)):
        img_clean = images[i].squeeze().detach().cpu().numpy()
        img_adv = adv_images[i].squeeze().detach().cpu().numpy()
        score = ssim(img_clean, img_adv, data_range=1.0)
        ssim_scores.append(score)
    avg_ssim = sum(ssim_scores) / len(ssim_scores)

    # Pełna ewaluacja
    acc = evaluate_under_attack(model, test_loader, attack)

    results.append({
        'epsilon': eps,
        'accuracy': acc,
        'ssim': avg_ssim,
        'time': end_time - start_time
    })



# Wywołanie wizualizacji
visualize_attack(model, test_loader, attack)

# Wykres strat i dokładności
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
