import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn as nn
from tqdm import tqdm
import argparse
import json

# ===== Argument Parser =====
parser = argparse.ArgumentParser(description="Evaluate a trained model on the test dataset.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (.pth file)")
parser.add_argument("--model_name", type=str, required=True, choices=[
    "efficientnet_b0", "densenet121", "resnet18", "mobilenet_v3_small"
], help="Model architecture used")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📊 Evaluating on: {device}")

# ===== Transform Test Data =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===== Dataset Path =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.join(BASE_DIR, "../dataset/test")
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

class_names = test_dataset.classes
num_classes = len(class_names)
print("📁 Classes:", class_names)

# ===== Model Loader =====
def load_model(model_name, model_path, num_classes):
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError("❌ Model tidak dikenali")

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

# ===== Load Model =====
model = load_model(args.model_name, args.model_path, num_classes)

# ===== Evaluate =====
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="🧪 Evaluating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ===== Report =====
print("\n📊 Classification Report:")
report = classification_report(all_labels, all_preds, target_names=class_names)
print(report)

# ===== Optional: Save report to file =====
report_path = f"classifier/eval_report_{args.model_name}.txt"
os.makedirs("classifier", exist_ok=True)
with open(report_path, "w") as f:
    f.write(report)
print(f"✅ Laporan evaluasi disimpan di {report_path}")