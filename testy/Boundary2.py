import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import foolbox as fb
import time
from skimage.metrics import structural_similarity as ssim



# 1) KONFIGURACJA I WCZYTANIE DANYCH
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


# 2) DEFINICJA MODELU CNN
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
criterion_ce = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 3) TRENING MODELU
for epoch in range(5):
    model.train()
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion_ce(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/5, Loss: {running_loss:.4f}")


# 4) TESTOWANIE MODELU
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


# 5) PRZYGOTOWANIE DO WIELOKROTNYCH ATACKÓW
NUM_EXAMPLES = 3

# 5.1) Owiń model w Foolboxa
fmodel = fb.PyTorchModel(model, bounds=(0, 1))

# 5.2) Przechowamy listy:
#    originals   – tensory oryginalnych, poprawnie sklasyfikowanych obrazów
#    orig_labels – odpowiadające etykiety
#    starts      – tensory “punktów startowych”, które model sklasyfikował inaczej niż każda oryginalna etykieta
originals   = []
orig_labels = []
starts      = []

# 5.3) Przeszukujemy zbiór testowy:
#       - gdy znajdziemy obraz sklasyfikowany poprawnie (pred == label) i nie przekroczyliśmy jeszcze NUM_EXAMPLES,
#         dodajemy go do “originals” i do “orig_labels”.
#       - dla każdego takiego obrazu wstępnie zapisujemy etykietę, ale punkt startowy (starts) znajdziemy dopiero
#         po zebraniu wszystkich “origingłów” (łatwiej w jednym cyklu).
# Najpierw zbierzemy N przykładów oryginalnych, potem w drugim etapie dla każdego znajdziemy punkt startowy.

print(f"--- Znajdowanie {NUM_EXAMPLES} przykładów oryginalnych (poprawnie sklasyfikowanych) ---")
with torch.no_grad():
    for img, lbl in testloader:
        img, lbl = img.to(device), lbl.to(device)
        preds = model(img).argmax(1)
        for i in range(img.shape[0]):
            if len(originals) < NUM_EXAMPLES and preds[i] == lbl[i]:
                originals.append(img[i : i+1].clone())
                orig_labels.append(lbl[i : i+1].clone())
                # Gdy już uzbieramy NUM_EXAMPLES, przerwijmy oba pętle
                if len(originals) == NUM_EXAMPLES:
                    break
        if len(originals) == NUM_EXAMPLES:
            break

if len(originals) < NUM_EXAMPLES:
    raise RuntimeError(f"Nie znaleziono {NUM_EXAMPLES} przykładów poprawnie sklasyfikowanych.")

# 5.4) Dla każdego oryginału znajdźmy punkt startowy:
print(f"--- Znajdowanie punktów startowych dla każdego z {NUM_EXAMPLES} przykładów ---")
for idx in range(NUM_EXAMPLES):
    orig_img   = originals[idx]
    orig_label = orig_labels[idx].item()
    # Przechodzimy po zbiorze testowym, żeby znaleźć obraz sklasyfikowany inaczej niż orig_label
    found = False
    with torch.no_grad():
        for img2, lbl2 in testloader:
            img2, lbl2 = img2.to(device), lbl2.to(device)
            preds2 = model(img2).argmax(1)
            for j in range(img2.shape[0]):
                if preds2[j].item() != orig_label:
                    starts.append(img2[j : j+1].clone())
                    found = True
                    break
            if found:
                break
    if not found:
        raise RuntimeError(f"Nie znaleziono punktu startowego dla oryginalnej etykiety {orig_label}.")

# Na tym etapie mamy:
#   originals   – lista len=NUM_EXAMPLES, każdy entry to tensor 1×1×28×28
#   orig_labels – lista len=NUM_EXAMPLES, każdy entry to tensor ([etykieta])
#   starts      – lista len=NUM_EXAMPLES, każdy entry to tensor 1×1×28×28

# 5.5) Przygotujmy kryteria do ataków – lista obiektów Misclassification
criteria_fb = [fb.criteria.Misclassification(lab) for lab in orig_labels]

# 5.6) Stwórzmy listę, w której przechowamy finalne “przykłady po ataku”
adversarials = []

# 5.7) Dla każdego oryginału uruchom BoundaryAttack i zapisz wynik:
print(f"--- Uruchamianie BoundaryAttack dla {NUM_EXAMPLES} przykładów ---")

start_time = time.time()

for idx in range(NUM_EXAMPLES):
    print(f"  Atak {idx+1}/{NUM_EXAMPLES} (etykieta oryginału = {orig_labels[idx].item()}) ...")
    attack = fb.attacks.BoundaryAttack()

    adv_ex = attack.run(
        fmodel,
        originals[idx],
        criteria_fb[idx],
        starting_points=starts[idx]
    )
    adversarials.append(adv_ex)

    # Oblicz SSIM
    orig_np = originals[idx].squeeze().cpu().numpy()
    adv_np  = adv_ex.squeeze().detach().cpu().numpy()
    ssim_val = ssim(orig_np, adv_np, data_range=1.0)  # Zakładamy zakres 0–1

    print(f"    → SSIM (oryginał vs adversarial): {ssim_val:.4f}")

end_time = time.time()

print(f"--- Gotowe! ---")
print(f"Całkowity czas generowania perturbacji dla {NUM_EXAMPLES} obrazów: {end_time - start_time:.2f} sekundy")


# 6) WIZUALIZACJA WIĘKSZEJ LICZBY PRZYKŁADÓW
# Stworzymy siatkę wykresów: NUM_EXAMPLES wierszy × 3 kolumn (oryginał / start / after-attack)
plt.figure(figsize=(3 * 3, NUM_EXAMPLES * 3))

for idx in range(NUM_EXAMPLES):

    # 6.1) Oryginał
    plt.subplot(NUM_EXAMPLES, 3, idx * 3 + 1)
    plt.imshow(originals[idx].squeeze().cpu(), cmap='gray')
    plt.title(f'Oryginalny\nlbl={orig_labels[idx].item()}')
    plt.axis('off')

    # 6.2) Punkt startowy
    plt.subplot(NUM_EXAMPLES, 3, idx * 3 + 2)
    plt.imshow(starts[idx].squeeze().cpu(), cmap='gray')
    with torch.no_grad():
        start_pred = model(starts[idx]).argmax(1).item()
    plt.title(f'Start\npred={start_pred}')
    plt.axis('off')

    # 6.3) Wynik po ataku
    plt.subplot(NUM_EXAMPLES, 3, idx * 3 + 3)
    plt.imshow(adversarials[idx].squeeze().detach().cpu(), cmap='gray')
    with torch.no_grad():
        final_pred = model(adversarials[idx].to(device)).argmax(1).item()
    plt.title(f'Po ataku\npred={final_pred}')
    plt.axis('off')

plt.tight_layout()
plt.show()
