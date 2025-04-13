import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import json

# =============================
# 🎛️ Pilihan Model
# =============================
available_models = {
    1: ("EfficientNet-B0", models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, "classifier[1]"),
    2: ("DenseNet121", models.densenet121, models.DenseNet121_Weights.DEFAULT, "classifier"),
    3: ("ResNet18", models.resnet18, models.ResNet18_Weights.DEFAULT, "fc"),
    4: ("MobileNetV3 Small", models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT, "classifier[3]"),
}

# =============================
# 📥 User Input
# =============================
print("📦 Pilih model yang ingin digunakan:")
for idx, (name, _, _, _) in available_models.items():
    print(f"{idx}. {name}")

model_choice = int(input("Masukkan angka model: ").strip())
epochs = int(input("Masukkan jumlah epoch: ").strip())
batch_size = int(input("Masukkan batch size: ").strip())

if model_choice not in available_models:
    raise ValueError("❌ Pilihan model tidak valid.")

model_name, model_fn, model_weight, classifier_attr = available_models[model_choice]

# =============================
# ⚙️ Device
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 Training on: {device} | Model: {model_name}")

# =============================
# 🧽 Transformasi
# =============================
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =============================
# 📂 Dataset & Loader
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(BASE_DIR, "../dataset/train")
test_dir = os.path.join(BASE_DIR, "../dataset/test")

print("📁 Train dir:", os.path.abspath(train_dir))  # Optional debug
print("📁 Test dir:", os.path.abspath(test_dir))    # Optional debug

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(test_dir, transform=val_transforms)

val_indices = list(range(len(val_dataset)))
with open("classifier/val_indices.pkl", "wb") as f:
    pickle.dump(val_indices, f)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# =============================
# 🧠 Load Model
# =============================
model = model_fn(weights=model_weight)
num_classes = len(train_dataset.classes)
print("📁 Detected classes:", train_dataset.classes)

# Ganti classifier-nya
classifier_layer = eval(f"model.{classifier_attr}")
in_features = classifier_layer.in_features
new_layer = nn.Linear(in_features, num_classes)
exec(f"model.{classifier_attr} = new_layer")

model.to(device)
print(f"✅ Loaded {model_name} dengan {num_classes} kelas.")

# =============================
# 🎯 Loss & Optimizer
# =============================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# =============================
# 📝 Inisialisasi History
# =============================
history = {
    "model_name": model_name,
    "num_classes": num_classes,
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": 1e-4,
    "train_loss": [],
    "train_acc": [],
    "val_acc": []
}

# =============================
# 🔁 Training Loop
# =============================
for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    loop = tqdm(train_loader, desc=f"🚂 Epoch {epoch+1}/{epochs}")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.detach().item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loop.set_postfix(loss=loss.detach().item())

    epoch_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    print(f"📚 Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {train_acc:.2f}%")

    # 🔍 Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total
    print(f"🔍 Validation Accuracy: {val_acc:.2f}%")

    # Simpan ke history
    history["train_loss"].append(epoch_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    # 💾 Save model
    os.makedirs("classifier/model", exist_ok=True)
    model_filename = f"letter_classifier_{model_name.lower().replace(' ', '_')}_e{epoch+1}_bs{batch_size}.pth"
    torch.save(model.state_dict(), os.path.join("classifier/model", model_filename))
    print(f"✅ Model disimpan ke classifier/model/{model_filename}")

# =============================
# ✅ Simpan History ke File
# =============================
history_filename = f"classifier/history_{model_name.lower().replace(' ', '_')}_e{epochs}_bs{batch_size}.json"
with open(history_filename, "w") as f:
    json.dump(history, f, indent=4)
print(f"📝 History disimpan ke {history_filename}")