import os
import json
import requests
import cv2
import numpy as np
from app.core.config import Config
from ultralytics import SAM

# Preprocessing
def segment_image(image_path, model, labels=None, device='cpu'):
    # Baca gambar
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    # Prediksi menggunakan model YOLO
    result = model.predict(image_path, points=[w / 2, h / 2], labels=labels or [1], device=device)

    # Akses masks
    masks = result[0].masks  # results adalah objek YOLO
    if masks is not None:
        mask_array = masks.data.cpu().numpy()  # Konversi ke NumPy array
    else:
        raise ValueError("No masks detected.")

    # Ambil gambar asli
    orig_img = result[0].orig_img

    # Ambil mask pertama (jika ada beberapa objek)
    selected_mask = mask_array[0]  # Misal ambil mask pertama

    # Konversi mask menjadi biner (0 dan 255 untuk aplikasi pada gambar)
    binary_mask = (selected_mask > 0.5).astype(np.uint8) * 255

    # Terapkan mask ke gambar asli
    masked_image = cv2.bitwise_and(orig_img, orig_img, mask=binary_mask)

    # Hitung bounding box
    x, y, w, h = cv2.boundingRect(binary_mask)

    # Potong area berdasarkan bounding box
    cropped_image = masked_image[y:y + h, x:x + w]
    cropped_mask = binary_mask[y:y + h, x:x + w]
    cropped_image = cropped_image.copy()

    # Ubah latar belakang hitam menjadi putih
    cropped_image[cropped_mask == 0] = [255, 255, 255]
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

    return cropped_image


# Run inference on an image
def predict_image(img_path):
    url = "https://predict.ultralytics.com"
    headers = {"x-api-key": Config.API_KEY}
    data = {"model": Config.MODEL_URL, "imgsz": 640, "conf": 0.85, "iou": 0.70}
    
    model = SAM('mobile_sam.pt')

    cropped_image = segment_image(img_path, model, labels=[1], device='cpu')
    file_path = "tmp/cropped_image.jpg"
    cv2.imwrite(file_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
    with open(file_path, "rb") as f:
        response = requests.post(url, headers=headers, data=data, files={"file": f})
    # delete temporary file
    os.remove(file_path)

    # Check for successful response
    response.raise_for_status()

    # Print inference results
    return response.json()