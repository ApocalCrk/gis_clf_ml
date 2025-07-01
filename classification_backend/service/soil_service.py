import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from typing import List, Dict, Any, Optional
from config import Config
from model.PlantRecommendation import PlantRecommendation

class SoilAnalysisService:
    def __init__(self):
        self.model = load_model(Config.MODEL_PATH)
        self.labels = Config.SOIL_LABELS
        self.scientific_to_common = Config.SCIENTIFIC_TO_COMMON
        self.cnn_to_texture_map = Config.CNN_TO_TEXTURE_MAP
    
    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess image for model prediction"""
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        img = cv2.resize(img, Config.IMAGE_SIZE)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    
    def predict_soil_texture(self, image_bytes: bytes) -> Dict[str, Any]:
        """Predict soil texture from image"""
        img = self.preprocess_image(image_bytes)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        prediction = self.model.predict(img_array)
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction))
        texture_label = self.labels[predicted_class]
        
        return {
            "predicted_class": predicted_class,
            "label": texture_label,
            "confidence": confidence
        }
    
    def get_confidence_warning(self, confidence: float) -> tuple[str, str]:
        """Get warning and high confidence messages based on confidence score"""
        warning = ""
        high_confidence = ""
        
        if confidence < 0.6:
            warning = "Peringatan: Ketepatan prediksi rendah. Silakan coba lagi dengan gambar yang lebih jelas atau coba gambar lain."
        elif confidence < 0.8:
            warning = "Peringatan: Ketepatan prediksi sedang. Silakan coba lagi dengan gambar yang lebih jelas atau coba gambar lain."
        else:
            high_confidence = "Ketepatan prediksi tinggi. Namun, tetap disarankan untuk memverifikasi hasil ini dengan ahli tanah atau menggunakan metode lain."
        
        return warning, high_confidence
    
    def get_recommendations_by_texture(self, plant_conditions: List[Dict], cnn_label: str) -> List[PlantRecommendation]:
        """Get plant recommendations based on soil texture"""
        texture_labels = self.cnn_to_texture_map.get(cnn_label, [])
        
        recommendations = []
        for plant in plant_conditions:
            if any(texture in plant['soil_texture'] for texture in texture_labels):
                recommendations.append(PlantRecommendation(
                    suitable_crops=plant['label'],
                    soil=plant['soil_texture']
                ))
        
        return recommendations
    
    def find_suitable_plants_by_parameters(self, all_plants: List[Dict], parameters: Dict[str, Any]) -> List[str]:
        """Find suitable plants based on soil parameters"""
        scientific_label = parameters.get('soil')
        cnn_label = self.scientific_to_common.get(scientific_label, None)
        
        if not cnn_label:
            return []
        
        texture_labels = self.cnn_to_texture_map.get(cnn_label, [])
        recommendations = []
        
        for plant in all_plants:
            conditions = [
                any(texture in plant['soil_texture'] for texture in texture_labels),
                plant['ph_min'] <= parameters['ph'] <= plant['ph_max'],
                plant['potassium_min'] <= parameters['cec'] <= plant['potassium_max'],
                plant['carbon_min'] <= parameters['carbon'] <= plant['carbon_max'],
                plant['nitrogen_min'] <= parameters['nitrogen'] <= plant['nitrogen_max']
            ]
            
            if all(conditions):
                recommendations.append(plant['label'])
        
        return recommendations
    
    def format_recommendation_reason(self, plants: List[str], soil_type: str) -> str:
        """Format recommendation reason text"""
        if len(plants) > 1:
            return f"Tanaman yang cocok untuk tanah {soil_type} meliputi: {', '.join(plants)}."
        elif len(plants) == 1:
            return f"Tanaman yang cocok untuk tanah {soil_type} adalah: {plants[0]}."
        else:
            return f"Tidak ada tanaman yang cocok untuk kondisi tanah {soil_type} dengan parameter yang diberikan."