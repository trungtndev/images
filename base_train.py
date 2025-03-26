import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src2.dataset.dataset import ImageDataset
from src2.model.cnn import CNNClassifier
from src2.model.swinv1 import SwinClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def validate(model, val_loader, criterion, epoch, epochs):
    val_loss, val_correct, val_total = 0, 0, 0
    model.eval()

    for images, labels in tqdm(val_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        val_loss += loss.item()
        _, predicted = outputs.max(1)
        val_total += labels.size(0)
        val_correct += predicted.eq(labels).sum().item()
    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Val Loss: {val_loss / len(val_loader):.4f} |"
        f"Val Acc: {100 * val_correct / val_total:.2f}%")


def train_one_epoch(model, train_loader, criterion, optimizer, epoch, epochs=10):
    model.train()
    train_loss, correct, total = 0, 0, 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print(labels.shape, outputs.shape)
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {train_loss / len(train_loader):.4f} | "
        f"Train Acc: {100 * correct / total:.2f}%")


def main():
    train_dataset = ImageDataset(
        "data/CASIA2",
        "train_df.pkl"
    )
    val_dataset = ImageDataset(
        "data/CASIA2",
        "val_df.pkl"
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    model = CNNClassifier(num_classes=2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0004,
        weight_decay=0.005,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    epochs = 10
    for epoch in range(epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, epoch=epoch,epochs=epochs)
        validate(model, val_loader, criterion, epoch=epoch,epochs=epochs)
        scheduler.step()    

if __name__ == "__main__":
    main()
