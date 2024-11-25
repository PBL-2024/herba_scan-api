from app.core import model_predict, yolo
from fastapi import FastAPI, File, UploadFile
import shutil
import os
from ultralytics import SAM,FastSAM

# model = SAM('mobile_sam.pt')
model = FastSAM('FastSAM-s.pt')
app = FastAPI()

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    # create folder if not exist
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    # check if the file is an image
    if file.content_type.split('/')[0] != 'image':
        return {"predict": "Invalid file type"}
    # Save the uploaded file to the tmp directory
    file_location = f"tmp/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Perform prediction
    predict = model_predict.predict_image_knn('', file_location)
    
    # Delete the file after prediction
    os.remove(file_location)
    
    return {"predict": predict}

# @app.post("/predict/keras/")
# async def predict_image(file: UploadFile = File(...)):
#     # create folder if not exist
#     if not os.path.exists('tmp'):
#         os.makedirs('tmp')
#     # check if the file is an image
#     if file.content_type.split('/')[0] != 'image':
#         return {"predict": "Invalid file type"}
#     # Save the uploaded file to the tmp directory
#     file_location = f"tmp/{file.filename}"
#     with open(file_location, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
    
#     # Perform prediction
#     predict = model_keras.predict(file_location,0.9)
    
#     # Delete the file after prediction
#     os.remove(file_location)
    
#     return predict

@app.post("/predict/yolo/")
async def predict_image(file: UploadFile = File(...)):
    # create folder if not exist
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    # check if the file is an image
    if file.content_type.split('/')[0] != 'image':
        return {"predict": "Invalid file type"}
    # Save the uploaded file to the tmp directory
    file_location = f"tmp/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Perform prediction
    predict = yolo.predict_image(file_location,model)
    
    os.remove(file_location)
    
    return predict