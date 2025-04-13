import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Model Loader ======
def load_model(model_name, num_classes, model_path):
    checkpoint = torch.load(model_path, map_location=device)

    # Jika file adalah dictionary dengan 'model_state_dict'
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        num_classes = checkpoint.get('num_classes', num_classes)
    else:
        state_dict = checkpoint

        # Deteksi jumlah kelas dari ukuran layer output
        for key in state_dict.keys():
            if "classifier" in key and "weight" in key:
                num_classes = state_dict[key].size(0)
                break
            elif "fc.weight" in key:  # untuk ResNet
                num_classes = state_dict[key].size(0)
                break

    # Buat model dengan output layer sesuai jumlah kelas
    if model_name == "efficientnet_b0" or "letter_classifier_efficientnet" in model_name:
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "densenet121" or "letter_classifier_densenet" in model_name:
        model = models.densenet121(weights=None)
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "resnet18" or "letter_classifier_resnet18" in model_name:
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "mobilenet_v3_small" or "letter_classifier_mobilenet_v3" in model_name:
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError("❌ Model tidak dikenal")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, num_classes

# ====== Transformasi Gambar ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ====== Fungsi Prediksi Huruf ======
def predict_letter(model, img_bgr, classes):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        return classes[predicted.item()]
