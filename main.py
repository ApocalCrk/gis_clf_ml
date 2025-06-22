from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import httpx # type: ignore
import html
import json
import asyncio
import re
from pydantic import BaseModel
from tensorflow.keras.models import load_model # type: ignore
import pymysql # type: ignore

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recommendation_plants = []

kondisi_tanaman = []

db = pymysql.connect(
    host="localhost",
    user="root",
    password="",
    database="gis_system",
)

def fetch_plants():
    cursor = db.cursor(pymysql.cursors.DictCursor)
    cursor.execute("SELECT * FROM kondisi_tanaman WHERE label != 'Tidak Ada'")
    rows = cursor.fetchall()
    cursor.close()
    return rows

@app.on_event("startup")
async def startup_event():
    global kondisi_tanaman
    kondisi_tanaman = fetch_plants()

model = load_model("model/clf_model_phase_test.h5")

labels = ['Tanah Aluvial', 'Tanah Hitam', 'Tanah Liat', 'Tanah Merah']

scientific_to_common = {
    "Cambisols": "Tanah Aluvial",
    "Andosols": "Tanah Hitam",
    "Arenosols": "Tanah Liat",
    "Podzols": "Tanah Merah",
    "Gleysols": "Tanah Aluvial",
    "Fluvisols": "Tanah Aluvial",
    "Vertisols": "Tanah Liat",
    "Luvisols": "Tanah Liat",
    "Chernozem": "Tanah Hitam",
    "Phaeozem": "Tanah Hitam",
    "Acrisol": "Tanah Merah",
    "Oxisol": "Tanah Merah",
    "Spodosol": "Tanah Merah",
    "Histosol": "Tanah Aluvial",
    "Regosol": "Tanah Aluvial",
    "Leptosol": "Tanah Liat",
    "Calcisol": "Tanah Liat",
    "Salinosol": "Tanah Liat",
    "Solonchak": "Tanah Liat"
}

recommendations = [
    {
        "soil": "Tanah Aluvial",
        "suitable_crops": [
            "Padi", "Jagung", "Kacang Hijau", "Pisang", "Semangka", "Pepaya",
            "Bawang Merah", "Bawang Putih", "Daun Bawang", "Cabai", 
            "Melon", "Kacang Kedelai", "Kacang Tanah"
        ],
        "reason": "Tanah Aluvial subur, memiliki kandungan hara tinggi dan baik untuk pertanian musiman seperti padi, jagung, kacang-kacangan, sayuran, dan buah-buahan tropis yang butuh banyak air."
    },
    {
        "soil": "Tanah Hitam",
        "suitable_crops": [
            "Padi", "Jagung", "Kacang Hijau", "Pisang", "Mangga", "Jeruk", "Pepaya", "Kopi", "Kubis", "Teh"
        ],
        "reason": "Tanah hitam kaya bahan organik dan memiliki struktur gembur sehingga cocok untuk berbagai tanaman pangan, buah-buahan, serta tanaman tahunan seperti kopi dan teh."
    },
    {
        "soil": "Tanah Liat",
        "suitable_crops": [
            "Padi", "Kacang Hijau", "Pisang", "Pepaya", "Kacang Kedelai", "Ubi Jalar"
        ],
        "reason": "Tanah liat menahan air dengan baik, cocok untuk tanaman yang membutuhkan kelembaban tinggi dan akar kuat, termasuk tanaman umbi dan kacang-kacangan."
    },
    {
        "soil": "Tanah Merah",
        "suitable_crops": [
            "Jagung", "Mangga", "Kopi", "Cabai", "Singkong", "Ubi Jalar"
        ],
        "reason": "Tanah merah kurang subur dan cepat kering, tetapi cocok untuk tanaman yang toleran terhadap kondisi kering dan tanah miskin hara seperti singkong dan cabai."
    }
]

def find_suitable_plants(all_plants, parameters):
    recommendations = []
    for tanaman in all_plants:
        if (tanaman['ph_min'] <= parameters['ph'] <= tanaman['ph_max'] or
            tanaman['potassium_min'] <= parameters['cec'] <= tanaman['potassium_max'] or 
            tanaman['carbon_min'] <= parameters['carbon'] <= tanaman['carbon_max'] or
            tanaman['nitrogen_min'] <= parameters['nitrogen'] <= tanaman['nitrogen_max']):
            recommendations.append(tanaman['label'])
    return recommendations

@app.post("/api/soil/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0 

    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class = int(np.argmax(prediction, axis=1)[0])
    
    confidence = float(np.max(prediction))
    label = labels[predicted_class]
    
    recomendation_data = {}
    
    for soil in recommendations:
        if soil["soil"] == label:
            recomendation_data = {
                "suitable_crops": soil["suitable_crops"],
                "reason": soil["reason"]
            }
            break
    
    warning = ""
    high_confidence = ""
    
    if confidence < 0.6:
        warning = "Peringatan: Ketepatan prediksi rendah. Silakan coba lagi dengan gambar yang lebih jelas atau coba gambar lain."
    elif confidence < 0.8:
        warning = "Peringatan: Ketepatan prediksi sedang. Silakan coba lagi dengan gambar yang lebih jelas atau coba gambar lain."
    else:
        high_confidence = "Ketepatan prediksi tinggi. Namun, tetap disarankan untuk memverifikasi hasil ini dengan ahli tanah atau menggunakan metode lain."

    return {
        "predicted_class": predicted_class,
        "label": label,
        "recommendation": recomendation_data,
        "confidence": float(np.max(prediction)),
        "warning": warning,
        "high_confidence": high_confidence,
    }

class analyzeRequest(BaseModel):
    ph: float
    soil: str
    carbon: float
    nitrogen: float
    cec: float
    
@app.post("/api/soil/analyze")
async def analyze(request: analyzeRequest):    
    recommendation_plants = find_suitable_plants(kondisi_tanaman, {
        "ph": request.ph,
        "cec": request.cec,
        "carbon": request.carbon,
        "nitrogen": request.nitrogen
    })
    
    recommendation_plants = list(set(recommendation_plants))
    if not recommendation_plants:
        recommendation_plants = ["Tidak ada tanaman yang cocok dengan kondisi tanah ini."]
        
    if request.soil:
        for soil in recommendations:
            if soil["soil"] == scientific_to_common.get(request.soil, request.soil):
                recommendation_plants = {
                    "suitable_crops": soil["suitable_crops"],
                    "reason": soil["reason"],
                    "soil_parameters": request,
                    "plants_by_condition": recommendation_plants
                }
                break
    else:
        recommendation_plants = {
            "suitable_crops": [],
            "reason": "",
            "soil_parameters": request,
            "plants_by_condition": recommendation_plants
        }
    
    return recommendation_plants

def convert_bold(text):
    return re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)

def convert_italic(text):
    return re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
  
def convert_list_items(text):
  return re.sub(r'^\*\s+(.*)', r'<li>\1</li>', text, flags=re.MULTILINE)

def format_response(text):
    text = convert_bold(text)
    text = html.escape(text)

    text = text.replace("&lt;strong&gt;", "<strong>").replace("&lt;/strong&gt;", "</strong>")
    text = convert_list_items(text)
    text = convert_italic(text)
    text = text.replace("\n\n", "<br><br>").replace("\n", "<br>")

    return text

class qRequest(BaseModel):
    plant: str
    leafColor: str
    stemCondition: str
    growth: str
    leafCondition: list[str]

url_ollama = "https://ollama.noturmine.my.id/api/generate"

@app.post("/api/health-assessment")
async def question(request: qRequest):
    question = (
        f"I'm growing {request.plant}. The leaves are {request.leafColor}, "
        f"the stem is {request.stemCondition}, and the growth appears to be {request.growth}. "
        f"The leaves are {', '.join(request.leafCondition)}. Based on these symptoms, please assess the "
        f"health condition of the plant. Explain possible causes or potential threats"
        f" short answer (150 characters) and don't make it too long."
        f" Please provide the answer in Indonesian."
        f"don't ask anything, just answer the question."
    )

    req = {
        "model": "gemma3:1b",
        # "model": "qwen3:0.6b",
        # "model": "mistral:7b-instruct-v0.2-q4_0",
        # "model": "tinyllama:1.1b",
        "prompt": question,
        "stream": True,
        "options": {
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 20,
            "max_new_tokens": 64,
            "num_ctx": 256,
            "use_cache": True,
            "use_mlock": False,
            "use_gpu": False,
            "use_fp16": True,
            "use_4bit": True,
            "use_8bit": False,
            "num_predict": 150,
            "num_threads": 4,
            "num_batch": 1
        }
    }

    full_response = ""

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url_ollama, json=req) as resp:
            async for line in resp.aiter_lines():
                if line.strip():
                    try:
                        data = json.loads(line)
                        full_response += data.get("response", "")
                    except json.JSONDecodeError:
                        continue
    
    formatted_response = format_response(full_response)
    
    return {
        "message": question,
        "response": formatted_response,
        "status_code": 200
    }