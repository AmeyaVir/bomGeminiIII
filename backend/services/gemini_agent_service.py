import os
import json
import requests
from typing import Optional, List, Dict
from dotenv import load_dotenv
import re

load_dotenv()

class GeminiAgentService:
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

    def _extract_json_from_markdown(self, text: str) -> str:
        """
        Extracts a JSON string from a Markdown code block.
        Returns an empty string if no JSON code block is found.
        """
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            return match.group(1)
        return text

    def _call_api(self, prompt: str, response_mime_type: Optional[str] = None, temperature: float = 0.2) -> requests.Response:
        """
        Internal helper to make a call to the external Gemini API gateway.
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        }
        
        # Pass response_mime_type in a nested generationConfig as per the API's documentation
        if response_mime_type:
            payload["generationConfig"] = {"responseMimeType": response_mime_type}

        try:
            response = requests.post(self.url, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API call failed: {e}")

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
            response = self._call_api(user_prompt, response_mime_type="application/json")
            raw_data = response.json()
            if 'choices' not in raw_data or not raw_data['choices']:
                print(f"API response missing 'choices': {raw_data}")
                return []
            extracted_text = raw_data['choices'][0]['message']['content']
            
            # Extract JSON string from markdown and then load it
            json_string = self._extract_json_from_markdown(extracted_text)
            parsed_data = json.loads(json_string)
            if isinstance(parsed_data, list):
                return parsed_data
            else:
                return []
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response from API: {e}")
            print(f"Raw API Response: {response.text}")
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
            response = self._call_api(user_prompt)
            if 'choices' not in response.json() or not response.json()['choices']:
                print(f"API response missing 'choices': {response.text}")
                return False
            extracted_text = response.json()['choices'][0]['message']['content']
            return extracted_text.strip().lower() == 'true'
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
            response = self._call_api(user_prompt, response_mime_type="application/json")
            if 'choices' not in response.json() or not response.json()['choices']:
                print(f"API response missing 'choices': {response.text}")
                return {'qty': '', 'uom': ''}
            extracted_text = response.json()['choices'][0]['message']['content']

            # Extract JSON string from markdown and then load it
            json_string = self._extract_json_from_markdown(extracted_text)
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response from API: {e}")
            print(f"Raw API Response: {response.text}")
            return {'qty': '', 'uom': ''}
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
            response = self._call_api(prompt, response_mime_type="application/json")
            if 'choices' not in response.json() or not response.json()['choices']:
                print(f"API response missing 'choices': {response.text}")
                return []
            extracted_text = response.json()['choices'][0]['message']['content']

            # Extract JSON string from markdown and then load it
            json_string = self._extract_json_from_markdown(extracted_text)
            standardized_data = json.loads(json_string)
            if isinstance(standardized_data, list):
                return standardized_data
            else:
                return []
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response from API: {e}")
            print(f"Raw API Response: {response.text}")
            return []
        except Exception as e:
            print(f"Error standardizing item master with LLM: {e}")
            return []
            
    def find_best_match(self, extracted_item: Dict, kb_items: List[Dict]) -> Optional[Dict]:
        """
        Uses the LLM to find the best matching item from a list of knowledge base items,
        considering fuzzy part number, semantic material name, and other metadata.
        
        Args:
            extracted_item: The item extracted from the new document.
            kb_items: A list of candidate items from the knowledge base.
            
        Returns:
            The best matching knowledge base item with a confidence score, or None.
        """
        
        prompt = f"""
        You are a highly accurate inventory matching agent. Your task is to find the single best match from a list of candidate items for a new item.
        
        The new item to match is:
        - Part Number: {extracted_item.get('part_number', 'N/A')}
        - Material Name: {extracted_item.get('material_name', 'N/A')}
        - Description: {extracted_item.get('description', 'N/A')}
        - Vendor Name: {extracted_item.get('vendor_name', 'N/A')}
        
        The list of candidate items from the knowledge base is:
        {json.dumps(kb_items, indent=2)}
        
        Rules for matching:
        1. Prioritize an exact or very close fuzzy match on 'part_number'.
        2. If part numbers are not a strong match, use 'material_name' and 'description' to find a strong semantic match.
        3. 'vendor_name' is an important secondary piece of information for confirmation.
        4. If a strong match is found, return the full details of the best matching item from the list.
        5. If no confident match (e.g., more than one ambiguous match or no match at all) is found, return an empty JSON object.
        6. The response must be a single, valid JSON object and nothing else. Do not include any explanation or additional text.
        
        Best matching item (or an empty object if no confident match):
        """
        
        try:
            response = self._call_api(prompt, response_mime_type="application/json", temperature=0.1)
            extracted_text = response.json()['choices'][0]['message']['content']
            json_string = self._extract_json_from_markdown(extracted_text)
            match_data = json.loads(json_string)
            
            # The LLM is instructed to return an empty object if no match.
            # We add a confidence score here based on LLM output.
            if match_data:
                match_data['confidence_score'] = 0.8
                return match_data
            return None
        
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response from API: {e}")
            print(f"Raw API Response: {response.text}")
            return None
        except Exception as e:
            print(f"Error calling Gemini API for match check: {e}")
            return None
