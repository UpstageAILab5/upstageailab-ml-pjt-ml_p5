from fastapi import FastAPI, UploadFile, File
from io import BytesIO

app = FastAPI() # uvicorn main:app --reload

# 라우터 설정
@app.get('/')
def root():
    return {"Hello":"World"}