import httpx
import json
import re
import html
from typing import Dict, Any
from config import Config
from model.PlantRecommendation import HealthAssessmentRequest

class PlantHealthService:
    def __init__(self):
        self.ollama_url = Config.OLLAMA_URL
        self.ollama_config = Config.OLLAMA_CONFIG
    
    def create_assessment_prompt(self, request: HealthAssessmentRequest) -> str:
        """Create assessment prompt for AI model"""
        return (
            f"I'm growing {request.plant}. The leaves are {request.leafColor}, "
            f"the stem is {request.stemCondition}, and the growth appears to be {request.growth}. "
            f"The leaves are {', '.join(request.leafCondition)}. Based on these symptoms, please assess the "
            f"health condition of the plant. Explain possible causes or potential threats"
            f" short answer (150 characters) and don't make it too long."
            f" Please provide the answer in Indonesian."
            f"don't ask anything, just answer the question."
        )
    
    async def get_ai_assessment(self, prompt: str) -> str:
        """Get AI assessment from Ollama API"""
        request_data = {
            **self.ollama_config,
            "prompt": prompt
        }
        
        full_response = ""
        
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", self.ollama_url, json=request_data) as response:
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            full_response += data.get("response", "")
                        except json.JSONDecodeError:
                            continue
        
        return full_response
    
    def format_response_text(self, text: str) -> str:
        """Format response text with HTML formatting"""
        # Convert markdown-style formatting to HTML
        text = self._convert_bold(text)
        text = html.escape(text)
        
        # Restore HTML tags that were escaped
        text = text.replace("&lt;strong&gt;", "<strong>").replace("&lt;/strong&gt;", "</strong>")
        text = self._convert_list_items(text)
        text = self._convert_italic(text)
        
        # Convert line breaks
        text = text.replace("\n\n", "<br><br>").replace("\n", "<br>")
        
        return text
    
    def _convert_bold(self, text: str) -> str:
        """Convert **text** to <strong>text</strong>"""
        return re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    
    def _convert_italic(self, text: str) -> str:
        """Convert *text* to <i>text</i>"""
        return re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    
    def _convert_list_items(self, text: str) -> str:
        """Convert * item to <li>item</li>"""
        return re.sub(r'^\*\s+(.*)', r'<li>\1</li>', text, flags=re.MULTILINE)
    
    async def assess_plant_health(self, request: HealthAssessmentRequest) -> Dict[str, Any]:
        """Assess plant health based on symptoms"""
        prompt = self.create_assessment_prompt(request)
        ai_response = await self.get_ai_assessment(prompt)
        formatted_response = self.format_response_text(ai_response)
        
        return {
            "message": prompt,
            "response": formatted_response,
            "status_code": 200
        }