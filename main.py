from app.core import config, model_predict
from fastapi import FastAPI, File, UploadFile
import shutil
import os

config = config.Config()

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