
# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import img_to_array

# def preprocess_image(image_path, img_size):
#     # Baca gambar
#     img = cv2.imread(image_path)
    
#     # Resize gambar sesuai dengan input model
#     img = cv2.resize(img, (img_size, img_size))
    
#     # Konversi gambar ke array dan normalisasi
#     img_array = img_to_array(img) / 255.0
    
#     # Tambahkan dimensi batch: (height, width, channels) -> (1, height, width, channels)
#     img_array = np.expand_dims(img_array, axis=0)
    
#     return img_array

# def load_model(model_path):
#     model = tf.keras.models.load_model(model_path)
#     return model

# def predict(img_path,threshold=0.5):
#     model = load_model('model-3-epoch.keras')
#     # Path ke gambar yang ingin diprediksi
#     image_path = img_path
#     # Preprocessing gambar (misalnya ukuran gambar input yang diharapkan model adalah 224x224)
#     img_size = 224
#     preprocessed_image = preprocess_image(image_path, img_size)

#     # Prediksi menggunakan model
#     predictions = model.predict(preprocessed_image)

#     # Ambil kelas dengan probabilitas tertinggi
#     predicted_class = np.argmax(predictions, axis=1)

#     class_names = ['Daun Jambu Biji',
#     'Daun Jeruk',
#     'Daun Kumis Kucing',
#     'Daun Kunyit',
#     'Daun Pandan',
#     'Daun Pepaya',
#     'Daun Sirih',
#     'Daun Sirsak',
#     'Daung Nangka',
#     'Lidah Buaya']
#     # Konversi indeks prediksi menjadi nama kelas
#     predicted_class_name = class_names[predicted_class[0]]
#     # Ambil confidence score tertinggi
#     conf = np.max(predictions)
#     result = (conf > threshold)

#     if result:
#         return predicted_class_name
#     else:
#         return "Not Predicted"
