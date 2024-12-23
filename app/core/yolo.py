import os
import json
from fastapi import status, HTTPException
import requests
import cv2
import numpy as np
from app.core.config import Config

# Preprocessing
def segment_image(image_path, model, labels=None, device='cpu'):
    # Baca gambar
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Prediksi menggunakan model YOLO
    try:
        result = model.predict(
            image_path,
            # points=[w / 2, h / 2],
            # labels=labels or [1],
            conf=0.65,
            iou=0.90,
            device=device,
            imgsz=640,
            # save=True
        )
    except Exception as e:
        return img
    
    if result is None:
        return img
    # Akses masks
    masks = result[0].masks  # results adalah objek YOLO
    if masks is None:
        return None
    mask_array = masks.data.cpu().numpy()  # Konversi ke NumPy array

    # Ambil gambar asli
    orig_img = result[0].orig_img

    # Cari mask dengan kontur terbesar
    max_area = 0
    largest_mask = None
    for mask in mask_array:
        # Konversi mask menjadi biner
        binary_mask = (mask > 0.5).astype(np.uint8) * 255

        # Hitung kontur dan area
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                largest_mask = binary_mask

    if largest_mask is None:
        return None

    largest_mask = cv2.resize(largest_mask, (w,h))
    
    # Hitung bounding box dari mask terbesar
    x, y, w, h = cv2.boundingRect(largest_mask)

    # Potong area berdasarkan bounding box
    cropped_image = orig_img[y:y + h, x:x + w]

    # Konversi ke RGB untuk ditampilkan dengan matplotlib
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

    return cropped_image


# Run inference on an image
def predict_image(img_path,model):
    url = "https://predict.ultralytics.com"
    headers = {"x-api-key": Config.API_KEY}
    data = {"model": Config.MODEL_URL, "imgsz": 640, "conf": 0.8, "iou": 0.70}
    cropped_image = segment_image(img_path, model, labels=[1], device='cpu')
    if cropped_image is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No object detected in the image")

    # upscale image to with 640px
    cropped_image = cv2.resize(cropped_image, (800, 800))
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
