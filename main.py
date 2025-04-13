import os
import cv2
from ultralytics import YOLO
from classifier.predict import load_model, predict_letter
from utils.draw_utils import draw_boxes  # Impor fungsi draw_boxes

# ========== Global Variables ==========

# Memilih model yang akan digunakan
model_directory = "./classifier/model"
available_models = os.listdir(model_directory)
print("Daftar model yang tersedia:")
for i, model_name in enumerate(available_models):
    print(f"{i+1}. {model_name}")

model_choice = int(input("Pilih model (nomor): ")) - 1
selected_model_name = available_models[model_choice]

# Menentukan path model
model_path = os.path.join(model_directory, selected_model_name)

# Nama model untuk load_model()
model_name = selected_model_name.split(".")[0]  # Mengambil nama model tanpa ekstensi

# Placeholder huruf A–Z tanpa J (24 huruf)
ALL_CLASSES = [chr(i) for i in range(65, 74)] + [chr(i) for i in range(75, 91)]

# Memuat model
model, num_classes = load_model(model_name, num_classes=len(ALL_CLASSES), model_path=model_path)

# Sinkronkan kelas sesuai jumlah kelas di model
CLASSES = ALL_CLASSES[:num_classes]

print(f"Memuat model {selected_model_name} dengan {num_classes} kelas...")
print(f"Daftar kelas yang digunakan: {CLASSES}")

# Load model deteksi tangan
hand_detector = YOLO('yolo_hand_detection/weights/hand_yolov8s.pt')

# Buka webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi objek (tangan) dengan YOLO
    results = hand_detector(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cropped_hand = frame[y1:y2, x1:x2]

        predicted_letter = predict_letter(model, cropped_hand, CLASSES)
        frame = draw_boxes(frame, x1, y1, x2, y2, predicted_letter, conf)

    cv2.imshow('Sign Language Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
