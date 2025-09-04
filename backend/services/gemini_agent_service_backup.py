import google.generativeai as genai
import os
import json
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class GeminiAgentService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def extract_all_items(self, document_content: str) -> list:
        """
        Uses the LLM to extract a raw list of all potential auxiliary items from the document.
        Returns a single, valid JSON array of objects.
        """
        user_prompt = f"""
        Analyze the following document content and extract a raw list of all auxiliary items (consumables, jigs, tools) mentioned.
        For each item, identify its material_name, part_number, qty, uom, and vendor_name.
        If a detail is not explicitly mentioned, return a blank string for that field.
        The output must be a single, valid JSON array of objects, with each object containing all of the required keys. Do not include any other text or formatting.
        
        Document Content:
        {document_content}
        """
        try:
            response = self.model.generate_content(
                user_prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                )
            )
            # Ensure the response is a valid JSON array
            raw_data = json.loads(response.text)
            if isinstance(raw_data, list):
                return raw_data
            else:
                return []
        except Exception as e:
            print(f"Error calling Gemini API for item extraction: {e}")
            return []

    def check_for_match(self, text_to_search: str, item_name: str, part_number: Optional[str] = None) -> bool:
        """
        Uses the LLM to check for a specific item name or part number match within a block of text.
        Returns True if a match is found, False otherwise.
        """
        user_prompt = f"""
        Does the following document text contain a reference to the item name "{item_name}"?
        The part number "{part_number}" might also be used.
        Respond with only 'True' or 'False'. Do not include any other text.

        Document text:
        {text_to_search}
        """
        try:
            response = self.model.generate_content(user_prompt)
            return response.text.strip().lower() == 'true'
        except Exception as e:
            print(f"Error calling Gemini API for match check: {e}")
            return False

    def extract_details(self, document_content: str, item_name: str) -> dict:
        """
        Uses the LLM to extract specific details like quantity and UoM for a given item name.
        """
        user_prompt = f"""
        From the following document content, extract the quantity (qty) and unit of measure (uom) for the item named "{item_name}".
        If a detail is not found, use a blank string.
        The output must be a valid JSON object with the keys "qty" and "uom".

        Document Content:
        {document_content}
        """
        try:
            response = self.model.generate_content(
                user_prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                )
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"Error calling Gemini API for detail extraction: {e}")
            return {'qty': '', 'uom': ''}
            
    def standardize_item_master(self, csv_content: str) -> list:
        """
        Uses LLM to standardize column headers in a CSV string to a predefined format.
        """
        standard_columns = ["material_name", "part_number", "description", "vendor_name", "uom"]
        prompt = f"""
        Given the following CSV content, standardize the column names to match a predefined list.
        Map any equivalent columns (e.g., 'Item Code' to 'part_number'). If a column has no equivalent, ignore it.
        The output must be a single, valid JSON array of objects, with each object containing the standardized keys.
        
        Standard Columns: {standard_columns}
        
        CSV Content:
        {csv_content}
        
        Standardized JSON Array:
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                )
            )
            standardized_data = json.loads(response.text)
            if isinstance(standardized_data, list):
                return standardized_data
            else:
                return []
        except Exception as e:
            print(f"Error standardizing item master with LLM: {e}")
            return []
