import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# =========================================================
# 1. Device
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================================================
# 2. Hyperparameters
# =========================================================

BATCH_SIZE = 512
LEARNING_RATE = 1e-3
EPOCHS = 5
NUM_CLASSES = 10

# =========================================================
# 3. Transforms
# Resize CIFAR-10 images from 32x32 -> 224x224
# =========================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# 4. Load CIFAR-10 Dataset
# =========================================================

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform
)

# =========================================================
# 5. Create Dataloaders
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

# =========================================================
# 6. Load Pretrained ResNet18
# =========================================================

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# =========================================================
# 7. Replace Final Classification Layer
# =========================================================

in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)

model = model.to(device)

# =========================================================
# 8. Loss and Optimizer
# =========================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================================================
# 9. Training Function
# =========================================================

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader)

    for images, labels in loop:

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stats
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        loop.set_description("Training")
        loop.set_postfix(
            loss=loss.item(),
            acc=accuracy
        )

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc

# =========================================================
# 10. Evaluation Function
# =========================================================

def evaluate(model, loader, criterion):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss = running_loss / len(loader)
    accuracy = 100 * correct / total

    return loss, accuracy

# =========================================================
# 11. Training Loop
# =========================================================

best_accuracy = 0.0

for epoch in range(EPOCHS):

    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion
    )

    val_loss, val_acc = evaluate(
        model,
        test_loader,
        criterion
    )

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Acc : {train_acc:.2f}%")

    print(f"Val Loss  : {val_loss:.4f}")
    print(f"Val Acc   : {val_acc:.2f}%")

    # =====================================================
    # 12. Save Best Model
    # =====================================================

    if val_acc > best_accuracy:

        best_accuracy = val_acc

        torch.save(
            model.state_dict(),
            "best_resnet18_cifar10.pth"
        )

        print("Best model saved!")

print("\nTraining Complete!")
print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
