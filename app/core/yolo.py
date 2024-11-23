import json
import requests
import cv2
import numpy as np
from app.core.config import Config

# Preprocessing
def preprocessing_image(img):
    src = cv2.imread(img, 1)
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    alpha = cv2.erode(alpha.copy(), None, iterations=3)
    alpha = cv2.dilate(alpha.copy(), None, iterations=5)
    kernel = np.ones((3, 3), np.uint8)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)
    b, g, r = cv2.split(src)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)

    contours, _ = cv2.findContours(alpha, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    selected = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(selected)
    cropped = dst[y:y+h, x:x+w]
    mask = alpha[y:y+h, x:x+w]

    return cropped, mask

# Run inference on an image
def predict_image(img_path):
    url = "https://predict.ultralytics.com"
    headers = {"x-api-key": Config.API_KEY}
    data = {"model": Config.MODEL_URL, "imgsz": 640, "conf": 0.85, "iou": 0.45}

    cropped_image, _ = preprocessing_image(img_path)
    _, img_encoded = cv2.imencode('.png', cropped_image)
    files = {"file": ("image.png", img_encoded.tobytes(), "image/png")}

    response = requests.post(url, headers=headers, data=data, files=files)

    # Check for successful response
    response.raise_for_status()

    # Print inference results
    return response.json()