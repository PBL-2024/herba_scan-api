#import library
import numpy as np
import os
import cv2
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from matplotlib import pyplot as plt


def load_model(model):
    model = joblib.load(model)
    return model

# proses terbaik untuk saat ini
import cv2
import numpy as np

def preprocessing_image(img_path):
    # Baca gambar
    src = cv2.imread(img_path)

    # Step 1: Konversi ke ruang warna HSV
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    # Step 2: Masking berdasarkan rentang warna hijau pada HSV
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Step 3: Operasi morfologi untuk membersihkan noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Step 4: Terapkan Otsu's threshold pada grayscale (jika diperlukan)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Gabungkan mask HSV dengan hasil Otsu (opsional tergantung performa)
    combined_mask = cv2.bitwise_and(mask, otsu_mask)

    # Step 5: Temukan kontur
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Pastikan kontur ditemukan
    if len(contours) > 0:
        # Pilih kontur terbesar
        largest_contour = max(contours, key=cv2.contourArea)

        # Dapatkan bounding box dari kontur terbesar
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop gambar asli berdasarkan bounding box
        cropped_image = src[y:y+h, x:x+w]

        # Crop mask berdasarkan bounding box
        cropped_mask = combined_mask[y:y+h, x:x+w]

        return cropped_image, cropped_mask
    else:
        print("No contours found.")
        return src, mask
    
# Ekstrak Fitur
hsv_properties  = ['hue', 'saturation', 'value']
glcm_properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
angles          = ['0', '45', '90', '135']
shape_properties= ['metric', 'eccentricity']

def extract_features(label_name,DIR,img):
  file_name = os.path.join(DIR, img)
  # Preprocessing
  cropped, mask = preprocessing_image(file_name)
  gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

  # Ekstraksi fitur HSV
  hsv_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
  image = hsv_image.reshape((hsv_image.shape[0] * hsv_image.shape[1], 3))
  clt = KMeans(n_clusters=3)
  labels = clt.fit_predict(image)
  label_counts = Counter(labels)
  dom_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

  # GLCM
  glcm = graycomatrix(gray,
                      distances=[5],
                      angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                      levels=256,
                      symmetric=True,
                      normed=True)

  glcm_props = [prop for name in glcm_properties for prop in graycoprops(glcm, name)[0]]

  # Ekstraksi fitur Shape
  label_img = label(mask)
  props = regionprops(label_img)
  eccentricity = getattr(props[0], 'eccentricity')
  area = getattr(props[0], 'area')
  perimeter = getattr(props[0], 'perimeter')
  if perimeter != 0:
    metric = (4 * np.pi * area) / (perimeter * perimeter)
  else:
    metric = 0

  fitur = [file_name, dom_color[0], dom_color[1], dom_color[2]] + glcm_props + [metric, eccentricity, label_name]
  return fitur

def predict_image_knn(DIR,img):
    # Load model dan LabelEncoder yang telah disimpan
    pkl = load_model('model_knn.pkl')
    scaler = pkl['scaler']
    label_encoder = pkl['label_encoder']
    model = pkl['model']
    # Ekstraksi fitur dari gambar baru
    features = extract_features('_',DIR,img)
    features = features[1:-1]

    features = scaler.transform([features])

    # Prediksi kelas
    predicted_label_encoded = model.predict(features)

    # Konversi hasil prediksi dari label numerik menjadi label asli
    predicted_label = label_encoder.inverse_transform([predicted_label_encoded][0])[0]

    return predicted_label