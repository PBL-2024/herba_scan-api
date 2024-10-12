from dotenv import load_dotenv
import os

class Config:
    load_dotenv(dotenv_path='.env')
    APP_NAME=os.getenv('APP_NAME')

    DB_HOST=os.getenv('DB_HOST')
    DB_PORT=os.getenv('DB_PORT')
    DB_USER=os.getenv('DB_USER')
    DB_PASS=os.getenv('DB_PASS')
    DB_NAME=os.getenv('DB_NAME')

    JWT_SECRET=os.getenv('JWT_SECRET')
    JWT_EXPIRES_IN=os.getenv('JWT_EXPIRES_IN')