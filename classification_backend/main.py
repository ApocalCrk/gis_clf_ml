from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import logging

from config.config import Config
from database.db import db_service
from model.PlantRecommendation import (
    SoilAnalyzeRequest, 
    HealthAssessmentRequest,
    SoilPredictionResponse,
    SoilAnalyzeResponse,
    HealthAssessmentResponse
)
from service.soil_service import SoilAnalysisService
from service.health_service import PlantHealthService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Agricultural Analysis API",
    description="API for soil analysis and plant health assessment",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
soil_service = SoilAnalysisService()
health_service = PlantHealthService()

# Global variable for plant conditions
plant_conditions: List[Dict[str, Any]] = []

@app.on_event("startup")
async def startup_event():
    """Initialize application data on startup"""
    global plant_conditions
    try:
        plant_conditions = db_service.fetch_plant_conditions()
        logger.info(f"Loaded {len(plant_conditions)} plant conditions from database")
    except Exception as e:
        logger.error(f"Failed to load plant conditions: {e}")
        plant_conditions = []

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    db_service.close()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Agricultural Analysis API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "plant_conditions_loaded": len(plant_conditions)
    }

@app.post("/api/soil/predict", response_model=SoilPredictionResponse)
async def predict_soil_texture(file: UploadFile = File(...)):
    """Predict soil texture from uploaded image"""
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Predict soil texture
        prediction_result = soil_service.predict_soil_texture(contents)
        
        # Get recommendations
        recommendations = soil_service.get_recommendations_by_texture(
            plant_conditions, 
            prediction_result['label']
        )
        
        # Format recommendations
        recommendation_data = {}
        if recommendations:
            suitable_crops = [rec.suitable_crops for rec in recommendations]
            reason = soil_service.format_recommendation_reason(
                suitable_crops, 
                prediction_result['label']
            )
            
            recommendation_data = {
                "suitable_crops": suitable_crops,
                "reason": reason
            }
        
        # Get confidence warnings
        warning, high_confidence = soil_service.get_confidence_warning(
            prediction_result['confidence']
        )
        
        return SoilPredictionResponse(
            predicted_class=prediction_result['predicted_class'],
            label=prediction_result['label'],
            recommendation=recommendation_data,
            confidence=prediction_result['confidence'],
            warning=warning,
            high_confidence=high_confidence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in soil prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/soil/analyze", response_model=SoilAnalyzeResponse)
async def analyze_soil_parameters(request: SoilAnalyzeRequest):
    """Analyze soil parameters and provide plant recommendations"""
    try:
        # Find suitable plants based on all parameters
        suitable_plants = soil_service.find_suitable_plants_by_parameters(
            plant_conditions, 
            {
                "ph": request.ph,
                "cec": request.cec,
                "carbon": request.carbon,
                "nitrogen": request.nitrogen,
                "soil": request.soil
            }
        )
        
        # Remove duplicates
        suitable_plants = list(set(suitable_plants))
        
        # Get soil-specific recommendations if soil type provided
        if request.soil:
            soil_recommendations = soil_service.get_recommendations_by_texture(
                plant_conditions, 
                Config.SCIENTIFIC_TO_COMMON.get(request.soil, "")
            )
            
            if soil_recommendations:
                suitable_plants = [rec.suitable_crops for rec in soil_recommendations]
            else:
                suitable_plants = []
        
        # Format reason
        reason = soil_service.format_recommendation_reason(suitable_plants, request.soil)
        
        return SoilAnalyzeResponse(
            suitable_crops=suitable_plants,
            reason=reason,
            soil_parameters=request,
            plants_by_condition=suitable_plants
        )
        
    except Exception as e:
        logger.error(f"Error in soil analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/health-assessment", response_model=HealthAssessmentResponse)
async def assess_plant_health(request: HealthAssessmentRequest):
    """Assess plant health based on symptoms"""
    try:
        assessment_result = await health_service.assess_plant_health(request)
        
        return HealthAssessmentResponse(
            message=assessment_result['message'],
            response=assessment_result['response'],
            status_code=assessment_result['status_code']
        )
        
    except Exception as e:
        logger.error(f"Error in health assessment: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)