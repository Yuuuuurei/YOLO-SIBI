import cv2

def draw_boxes(frame, x1, y1, x2, y2, label, conf):
    """
    Fungsi untuk menggambar bounding box dan label pada frame.
    frame: Gambar input (frame dari webcam)
    x1, y1, x2, y2: Koordinat bounding box
    label: Label dari deteksi objek (misalnya 'A', 'B', dll)
    conf: Confidence score dari deteksi
    """
    color = (0, 255, 0)  # Warna hijau untuk bounding box
    thickness = 2  # Ketebalan garis box

    # Gambar bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Tampilkan label dan confidence di bounding box
    label_text = f'{label} ({conf*100:.1f}%)'
    # Draw outline (white or black stroke)
    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 0), 1, cv2.LINE_AA)


    return frame
