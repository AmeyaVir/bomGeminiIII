import os
import json
import requests
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class TranslationService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        
        self.url = os.getenv("GEMINI_API_URL", "https://api.ai-gateway.tigeranalytics.com/chat/completions")
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _call_api(self, prompt: str) -> requests.Response:
        """
        Internal helper to make a call to the external Gemini API gateway for translation.
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            response = requests.post(self.url, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API call failed: {e}")

    def translate_to_english(self, text: str) -> str:
        """Translates Japanese text to English using Gemini API."""
        prompt = f"""
        Translate the following text from Japanese to English. Maintain all original formatting, including line breaks and tables.
        
        Japanese Text:
        {text}
        
        English Translation:
        """
        try:
            response = self._call_api(prompt)
            extracted_text = response.json()['choices'][0]['message']['content']
            return extracted_text
        except Exception as e:
            print(f"Error calling Gemini API for translation: {e}")
            return text
