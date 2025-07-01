import os
from typing import Dict, List

class Config:
    # Database Configuration
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_USER = os.getenv("DB_USER", "root")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_NAME = os.getenv("DB_NAME", "gis_system")
    
    # Model Configuration
    MODEL_PATH = "model/soil_texture_classifier.h5"
    IMAGE_SIZE = (224, 224)
    
    # API Configuration
    OLLAMA_URL = os.getenv("OLLAMA_URL", "https://ollama.noturmine.my.id/api/generate")
    
    # CORS Configuration
    CORS_ORIGINS = ["*"]
    
    # Soil Classification Labels
    SOIL_LABELS = [
        "clay",
        "sandy", 
        "loamy",
        "laterite",
        "humus"
    ]
    
    # Scientific to Common Name Mapping
    SCIENTIFIC_TO_COMMON: Dict[str, str] = {
        "Vertisols": "clay",
        "Planosols": "clay",
        "Luvisols": "clay",
        "Nitisols": "clay",
        "Regosols": "sandy",
        "Arenosols": "sandy",
        "Leptosols": "sandy",
        "Cambisols": "loamy",
        "Fluvisols": "loamy",
        "Gleysols": "loamy",
        "Acrisols": "laterite",
        "Ferralsols": "laterite",
        "Lixisols": "laterite",
        "Histosols": "humus",
        "Andosols": "humus",
        "Umbrisols": "humus"
    }
    
    # CNN to Texture Mapping
    CNN_TO_TEXTURE_MAP: Dict[str, List[str]] = {
        "clay": ["Clay Soil", "Clay Loam", "Silty Clay"],
        "sandy": ["Sandy Soil", "Sandy Loam"],
        "loamy": ["Loam", "Silty Loam"],
        "laterite": ["Red Soil", "Laterite Soil"],
        "humus": ["Peaty Soil", "Organic Soil", "Humus"]
    }
    
    # Ollama Model Configuration
    OLLAMA_CONFIG = {
        "model": "gemma3:1b",
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