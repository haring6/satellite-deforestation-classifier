"""
EuroSAT Land Classifier - Local GPU training script
Optimized for laptop GPUs with limited VRAM (e.g. RTX 3050 6GB) via mixed precision.

Run with:  python train_local.py
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms, models
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

BATCH_SIZE = 16          # reduced for 6GB VRAM
NUM_WORKERS = 4
HEAD_EPOCHS = 10
FINETUNE_EPOCHS = 8


class EuroSATWrapper(Dataset):
    def __init__(self, hf_dataset, transform):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert('RGB')
        label = item['label']
        return self.transform(image), label


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with autocast(device_type='cuda'):
                outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def train_epochs(model, train_loader, val_loader, num_epochs, optimizer,
                  criterion, scaler, history, device, tag):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        t0 = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        val_acc = evaluate(model, val_loader, device)
        avg_loss = running_loss / len(train_loader)
        history['loss'].append(avg_loss)
        history['val_acc'].append(val_acc)
        dt = time.time() - t0
        print(f"[{tag}] Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - "
              f"Val Acc: {val_acc:.4f} - {dt:.1f}s")


def main():
    assert torch.cuda.is_available(), "CUDA not available - check your driver/PyTorch install"
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    print("\nDownloading/loading EuroSAT (one-time download, ~90MB)...")
    ds = load_dataset("blanchon/EuroSAT_RGB")
    class_names = ds['train'].features['label'].names
    print("Classes:", class_names)
    print("Total images:", len(ds['train']))

    split1 = ds['train'].train_test_split(test_size=0.2, seed=42)
    train_data = split1['train']
    split2 = split1['test'].train_test_split(test_size=0.5, seed=42)
    val_data = split2['train']
    test_data = split2['test']
    print(f"Train: {len(train_data)}  Val: {len(val_data)}  Test: {len(test_data)}")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_set = EuroSATWrapper(train_data, train_transform)
    val_set = EuroSATWrapper(val_data, eval_transform)
    test_set = EuroSATWrapper(test_data, eval_transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    model = models.resnet50(weights='IMAGENET1K_V2')
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(device='cuda')
    history = {'loss': [], 'val_acc': []}

    print("\n--- Stage 1: training final layer only ---")
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    train_epochs(model, train_loader, val_loader, HEAD_EPOCHS, optimizer,
                 criterion, scaler, history, device, "head")

    print("\n--- Stage 2: fine-tuning full model ---")
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train_epochs(model, train_loader, val_loader, FINETUNE_EPOCHS, optimizer,
                 criterion, scaler, history, device, "finetune")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'])
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("Saved training_curves.png")

    print("\n--- Final test set evaluation ---")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            with autocast(device_type='cuda'):
                outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)
    with open('classification_report.txt', 'w') as f:
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - EuroSAT Test Set')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Saved confusion_matrix.png")

    torch.save(model.state_dict(), 'eurosat_resnet50.pth')
    print("\nSaved model to eurosat_resnet50.pth")
    print("\nDone. Send back: the printed classification_report, training_curves.png, and confusion_matrix.png")


if __name__ == '__main__':
    main()