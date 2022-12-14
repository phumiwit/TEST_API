from fastapi import FastAPI, File, UploadFile

import numpy as np
import librosa
import math
from pydub import AudioSegment
import math
import os
import pickle
from keyword_spotting_service import Keyword_Spotting_Service
import uvicorn

app = FastAPI()
# pickle_in = open("Music_genre_CNN.pkl","rb")
# CNN = pickle.load(pickle_in)

@app.get('/')
def Hello():
    return {'Hello':'Hello'}

@app.get('/predict')
def predict_genre(path : str):
    kss = Keyword_Spotting_Service()
    keyword1,keyword2= kss.prediction(path)
    return {"prediction":keyword1}

@app.get('/value')
def value_genre(path : str):
    kss = Keyword_Spotting_Service()
    keyword1,keyword2 = kss.prediction(path)
    
    return keyword2
if __name__ == "__main__":
    uvicorn.run(app,host='127.0.0.1',port=8000)
