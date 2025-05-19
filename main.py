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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model("model/clf_model_phase_test.h5")

labels = ['Tanah Aluvial', 'Tanah Hitam', 'Tanah Liat', 'Tanah Merah']

# Database tanaman dan kebutuhan dasarnya
plants = [
    {"name": "padi", "min_ph": 5.0, "max_ph": 6.5, "min_organic": 20, "min_water": 5, "soil": ["Cambisols", "Fluvisols"]},
    {"name": "jagung", "min_ph": 5.5, "max_ph": 7.0, "min_organic": 20, "min_water": 4, "soil": ["Cambisols", "Andisols"]},
    {"name": "kacang arab", "min_ph": 5.0, "max_ph": 7.0, "min_organic": 15, "min_water": 4, "soil": ["Cambisols"]},
    {"name": "kacang merah", "min_ph": 5.5, "max_ph": 6.8, "min_organic": 20, "min_water": 4, "soil": ["Cambisols", "Luvisols"]},
    {"name": "kacang gude", "min_ph": 5.5, "max_ph": 7.0, "min_organic": 15, "min_water": 3, "soil": ["Vertisols", "Cambisols"]},
    {"name": "kacang ngengat", "min_ph": 6.0, "max_ph": 7.5, "min_organic": 10, "min_water": 3, "soil": ["Cambisols"]},
    {"name": "kacang hijau", "min_ph": 5.0, "max_ph": 6.5, "min_organic": 10, "min_water": 3, "soil": ["Cambisols", "Fluvisols"]},
    {"name": "kacang hitam", "min_ph": 5.5, "max_ph": 6.5, "min_organic": 20, "min_water": 3, "soil": ["Cambisols"]},
    {"name": "kacang lentil", "min_ph": 6.0, "max_ph": 7.5, "min_organic": 15, "min_water": 3, "soil": ["Cambisols"]},
    {"name": "delima", "min_ph": 5.5, "max_ph": 7.0, "min_organic": 20, "min_water": 4, "soil": ["Cambisols"]},
    {"name": "pisang", "min_ph": 5.5, "max_ph": 6.5, "min_organic": 25, "min_water": 5, "soil": ["Cambisols", "Andisols"]},
    {"name": "mangga", "min_ph": 5.5, "max_ph": 7.5, "min_organic": 15, "min_water": 4, "soil": ["Cambisols", "Luvisols"]},
    {"name": "anggur", "min_ph": 6.0, "max_ph": 7.5, "min_organic": 10, "min_water": 3, "soil": ["Cambisols"]},
    {"name": "semangka", "min_ph": 6.0, "max_ph": 6.8, "min_organic": 15, "min_water": 4, "soil": ["Cambisols"]},
    {"name": "blewah", "min_ph": 6.0, "max_ph": 6.8, "min_organic": 15, "min_water": 4, "soil": ["Cambisols"]},
    {"name": "apel", "min_ph": 6.0, "max_ph": 6.8, "min_organic": 20, "min_water": 5, "soil": ["Luvisols"]},
    {"name": "jeruk", "min_ph": 6.0, "max_ph": 6.5, "min_organic": 25, "min_water": 4, "soil": ["Cambisols"]},
    {"name": "pepaya", "min_ph": 5.5, "max_ph": 6.7, "min_organic": 20, "min_water": 4, "soil": ["Cambisols"]},
    {"name": "kelapa", "min_ph": 5.0, "max_ph": 7.0, "min_organic": 20, "min_water": 4, "soil": ["Cambisols"]},
    {"name": "kapas", "min_ph": 6.0, "max_ph": 7.5, "min_organic": 10, "min_water": 3, "soil": ["Cambisols"]},
    {"name": "rami", "min_ph": 6.0, "max_ph": 7.0, "min_organic": 15, "min_water": 5, "soil": ["Cambisols"]},
    {"name": "kopi", "min_ph": 5.0, "max_ph": 6.5, "min_organic": 25, "min_water": 4, "soil": ["Cambisols", "Andisols"]}
]

# Data tambahan berdasarkan jenis tanah umum
recommendations = [
    {
        "soil": "Tanah Aluvial",
        "suitable_crops": [
            "padi", "jagung", "kacang hijau", "kacang merah", "kacang hitam",
            "kacang gude", "pisang", "semangka", "blewah", "pepaya"
        ],
        "reason": "Tanah Aluvial subur, memiliki kandungan hara tinggi dan baik untuk pertanian musiman seperti padi, jagung, kacang-kacangan, dan buah-buahan tropis yang butuh banyak air."
    },
    {
        "soil": "Tanah Hitam",
        "suitable_crops": [
            "padi", "jagung", "kacang hijau", "kacang merah", "kacang arab", "pisang",
            "mangga", "anggur", "apel", "jeruk", "pepaya", "kelapa", "kopi"
        ],
        "reason": "Tanah hitam kaya bahan organik dan memiliki struktur gembur sehingga cocok untuk berbagai tanaman pangan dan buah-buahan, serta tanaman tahunan seperti kopi dan kelapa."
    },
    {
        "soil": "Tanah Liat",
        "suitable_crops": [
            "padi", "kacang hijau", "kacang merah", "kacang hitam", "kacang arab",
            "pisang", "pepaya", "kapas", "rami"
        ],
        "reason": "Tanah liat menahan air dengan baik, cocok untuk tanaman yang membutuhkan kelembaban tinggi dan akar kuat, tetapi harus dikelola agar tidak terlalu padat."
    },
    {
        "soil": "Tanah Merah",
        "suitable_crops": [
            "jagung", "kacang gude", "kacang ngengat", "kacang lentil", "mangga",
            "anggur", "delima", "kapas", "rami", "kopi"
        ],
        "reason": "Tanah merah kurang subur dan cepat kering, tetapi cocok untuk tanaman yang toleran terhadap kondisi kering dan tanah kurang organik."
    }
]

def is_suitable(plant, request):
    return (
        plant["min_ph"] <= request.ph <= plant["max_ph"] and
        request.organic_matter >= plant["min_organic"] and
        request.water_content >= plant["min_water"] and
        request.soil in plant["soil"]
    )

soil_to_scientific = {
    "Tanah Aluvial": "Fluvisols",
    "Tanah Hitam": "Andisols",
    "Tanah Liat": "Vertisols",
    "Tanah Merah": "Cambisols"
}

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

class WaterQualityRequest(BaseModel):
    ph: float
    soil: str
    organic_matter: float
    water_content: float

@app.post("/api/soil/analyze")
async def analyze(request: WaterQualityRequest):
    """
    Recommends suitable plants based on soil parameters.r
    """
    # Find suitable plants based on scientific parameters
    recommended = [plant["name"] for plant in plants if is_suitable(plant, request)]
    
    response = {
        "soil_parameters": {
            "ph": request.ph,
            "organic_matter": request.organic_matter,
            "soil_class": request.soil,
            "water_content": request.water_content
        },
        "recommended_plants": recommended
    }
    
    return response

url_ollama = "https://ollama.noturmine.my.id/api/generate"

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
        "model": "gemma3:1b",  # Consider even smaller models (e.g., TinyLlama, Phi-2)
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