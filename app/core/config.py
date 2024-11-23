from dotenv import load_dotenv
import os

class Config:
    load_dotenv(dotenv_path='.env')
    API_KEY=os.getenv('API_KEY')
    MODEL_URL=os.getenv('MODEL_URL')