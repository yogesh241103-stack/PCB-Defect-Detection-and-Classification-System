import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm

# --- Configuration ---
DATA_DIR = "./extracted_rois"
MODEL_SAVE_PATH = "efficientnet_pcb_model.pth"
BATCH_SIZE = 32
MAX_EPOCHS = 30 # Early stopping will likely halt it sooner
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 5 # Stops if no improvement for 5 epochs

# --- 1. GPU Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# --- 2. Data Preprocessing & Augmentation ---
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Handles lighting variations
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- 3. Load Dataset ---
full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms['train'])
class_names = full_dataset.classes
print(f"Defect Classes Found: {class_names}")

# Split 80% Train, 20% Val
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Ensure validation set doesn't get random augmentations
val_dataset.dataset.transform = data_transforms['val']

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 4. Build EfficientNet Model ---
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# Anti-overfitting: Add Dropout before the final classification layer
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4, inplace=True),
    nn.Linear(num_ftrs, len(class_names))
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
# AdamW adds weight decay to prevent the model from memorizing the data
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
# Reduces learning rate automatically if accuracy plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

# --- 5. Training Loop ---
best_acc = 0.0
epochs_no_improve = 0
train_losses, val_losses = [], []
train_accs, val_accs = [], []

print("\nStarting Optimized Training...")
for epoch in range(MAX_EPOCHS):
    model.train()
    running_loss, running_corrects = 0.0, 0
    
    # Training pass
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Train]"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
    epoch_loss = running_loss / train_size
    epoch_acc = running_corrects.double() / train_size
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc.item())

    # Validation pass
    model.eval()
    val_loss, val_corrects = 0.0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Val]"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    val_epoch_loss = val_loss / val_size
    val_epoch_acc = val_corrects.double() / val_size
    val_losses.append(val_epoch_loss)
    val_accs.append(val_epoch_acc.item())
    
    scheduler.step(val_epoch_acc)
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f} | LR: {current_lr:.6f}")

    # Save best model and check early stopping
    if val_epoch_acc > best_acc:
        best_acc = val_epoch_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        epochs_no_improve = 0 
    else:
        epochs_no_improve += 1
        
    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print(f"\n[!] Early stopping triggered. Validation accuracy hasn't improved in {EARLY_STOPPING_PATIENCE} epochs.")
        break

print(f"\nTraining Complete. Best Validation Accuracy: {best_acc:.4f}")

# --- 6. Evaluation & Visualizations ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()
plt.savefig('training_metrics.png')

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.ylabel('Actual Defect')
plt.xlabel('Predicted Defect')
plt.savefig('confusion_matrix.png')

print("Evaluation plots saved as 'training_metrics.png' and 'confusion_matrix.png'")