import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
DATA_DIR = "./extracted_rois"
MODEL_PATH = "efficientnet_pcb_model.pth"
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Load Data (Validation Split) ---
test_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(DATA_DIR, transform=test_transforms)
class_names = full_dataset.classes

# We must use the same split seed or logic used in training to get the 20% validation set
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
_, val_dataset = random_split(full_dataset, [train_size, val_size], torch.Generator().manual_seed(42))

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 2. Load the Saved Model ---
def load_saved_model():
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_ftrs, len(class_names))
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(device)
    model.eval()
    return model

# --- 3. Run Evaluation ---
print(f"Testing model on {len(val_dataset)} images...")
model = load_saved_model()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# --- 4. Accuracy & Classification Report ---
print("\n" + "="*30)
print("FINAL TEST ACCURACY REPORT")
print("="*30)
print(classification_report(all_labels, all_preds, target_names=class_names))

# --- 5. Visual Confusion Matrix ---
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=class_names, yticklabels=class_names)
plt.title("Final Accuracy Confusion Matrix")
plt.ylabel('True Defect Type')
plt.xlabel('AI Predicted Type')
plt.savefig('final_test_results.png')
print("Detailed results saved as 'final_test_results.png'")