from typing import Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict
import json
import os
import logging
import requests
from dotenv import load_dotenv
import re


load_dotenv()


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")

class DocumentClassification(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    format: str = Field(description="The format of the document (PDF, JSON, EMAIL, or TEXT)")
    intent: str = Field(description="The intent of the document (INVOICE, RFQ, COMPLAINT, REGULATION, or OTHER)")
    confidence: float = Field(description="Confidence score of the classification")

class JsonAnalysis(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    missing_fields: List[str] = Field(description="List of missing fields in the JSON")
    anomalies: List[str] = Field(description="List of anomalies found in the JSON")
    suggested_schema: Dict[str, str] = Field(description="Suggested schema for the JSON")

class EmailAnalysis(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    sender: str = Field(description="Email sender address")
    recipient: str = Field(description="Email recipient address")
    subject: str = Field(description="Email subject")
    urgency: str = Field(description="Urgency level (HIGH, MEDIUM, or LOW)")
    key_points: List[str] = Field(description="Key points from the email")
    action_items: List[str] = Field(description="Action items from the email")
    sentiment: str = Field(description="Email sentiment (POSITIVE, NEUTRAL, or NEGATIVE)")

class TextAnalysis(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    summary: str = Field(description="Summary of the text content")
    key_topics: List[str] = Field(description="Main topics discussed in the text")
    sentiment: str = Field(description="Overall sentiment (POSITIVE, NEUTRAL, or NEGATIVE)")
    action_items: List[str] = Field(description="Action items or important points from the text")

class LangChainAgent:
    def __init__(self):
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GOOGLE_API_KEY}"
        self.headers = {
            "Content-Type": "application/json"
        }
    
    def _call_api(self, prompt: str) -> str:
        """Make API call to Google Gemini."""
        try:
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.3,
                    "topK": 1,
                    "topP": 1,
                    "maxOutputTokens": 2048,
                    "stopSequences": []
                },
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    }
                ]
            }
            
            response = requests.post(
                url=self.api_url,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            
           
            logging.debug(f"API Response: {response.text}")
            
            response_data = response.json()
            
            if not response_data.get("candidates"):
                raise Exception("No candidates in API response")
            
            candidate = response_data["candidates"][0]
            if not candidate.get("content") or not candidate["content"].get("parts"):
                raise Exception("No content in API response")
                
            content = candidate["content"]["parts"][0]["text"]
            if not content:
                raise Exception("Empty content in API response")
                
            return content
            
        except requests.exceptions.RequestException as e:
            logging.error(f"API call failed: {str(e)}")
            raise Exception(f"API call failed: {str(e)}")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse API response: {str(e)}")
            raise Exception(f"Failed to parse API response: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error in API call: {str(e)}")
            raise Exception(f"Unexpected error in API call: {str(e)}")
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response with error handling, removing markdown."""
        
        match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
        if match:
            json_string = match.group(1)
        else:
            
            json_string = response
            
        try:
           
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {str(e)}")
            logging.error(f"Raw response: {response}")
            raise Exception(f"Failed to parse JSON response: {str(e)}")
    
    def _create_classifier_prompt(self, content: str) -> str:
        """Create prompt for document classification."""
        return f"""Analyze the following content and determine its format and intent. You must respond with a valid JSON object in the following format:
{{
    "format": "PDF/JSON/EMAIL/TEXT",
    "intent": "INVOICE/RFQ/COMPLAINT/REGULATION/OTHER",
    "confidence": 0.0 to 1.0
}}

Content:
{content}

Remember to respond with ONLY the JSON object, no additional text."""
    
    def _create_json_prompt(self, content: str) -> str:
        """Create prompt for JSON analysis."""
        return f"""Analyze the following JSON structure and identify any missing or anomalous fields. You must respond with a valid JSON object in the following format:
{{
    "missing_fields": ["list", "of", "missing", "fields"],
    "anomalies": ["list", "of", "anomalies"],
    "suggested_schema": {{"field": "type"}}
}}

JSON Content:
{content}

Remember to respond with ONLY the JSON object, no additional text."""
    
    def _create_email_prompt(self, content: str) -> str:
        """Create prompt for email analysis."""
        return f"""Analyze the following email content and extract key information. You must respond with a valid JSON object in the following format:
{{
    "sender": "email sender",
    "recipient": "email recipient",
    "subject": "email subject",
    "urgency": "HIGH/MEDIUM/LOW",
    "key_points": ["list", "of", "key", "points"],
    "action_items": ["list", "of", "action", "items"],
    "sentiment": "POSITIVE/NEUTRAL/NEGATIVE"
}}

Email Content:
{content}

Remember to respond with ONLY the JSON object, no additional text."""
    
    def _create_text_prompt(self, content: str) -> str:
        """Create prompt for text analysis."""
        return f"""Analyze the following text content and extract key information. You must respond with a valid JSON object in the following format:
{{
    "summary": "text summary",
    "key_topics": ["list", "of", "topics"],
    "sentiment": "POSITIVE/NEUTRAL/NEGATIVE",
    "action_items": ["list", "of", "action", "items"]
}}

Text Content:
{content}

Remember to respond with ONLY the JSON object, no additional text."""
    
    def _create_pdf_prompt(self, content: str) -> str:
        """Create a prompt for analyzing PDF content."""
        return f"""Analyze the following PDF content and extract key information. You must respond with a valid JSON object in the following format:
{{
    "title": "Document title if available",
    "summary": "Brief summary of the document content",
    "key_points": ["List of main points or findings"],
    "metadata": {{
        "page_count": "Number of pages if available",
        "creation_date": "Document creation date if available",
        "author": "Document author if available"
    }},
    "entities": {{
        "organizations": ["List of organizations mentioned"],
        "people": ["List of people mentioned"],
        "dates": ["List of important dates"],
        "amounts": ["List of monetary amounts or quantities"]
    }}
}}

PDF Content:
{content}

Remember to respond with ONLY the JSON object, no additional text. If any field is not applicable or information is not available, use null or an empty array as appropriate."""
    
    def process_document(self, doc: Any) -> Dict[str, Any]:
        """Process a document using appropriate chain."""
        try:
           
            classifier_prompt = self._create_classifier_prompt(doc.page_content)
            classifier_response = self._call_api(classifier_prompt)
            classification = self._parse_json_response(classifier_response)
            
            logging.info(f"Document classified as: {classification['format']} with intent: {classification['intent']}")
            
            
            if classification["format"] == "JSON":
                json_prompt = self._create_json_prompt(doc.page_content)
                json_response = self._call_api(json_prompt)
                analysis = self._parse_json_response(json_response)
                return {
                    "format": classification["format"],
                    "intent": classification["intent"],
                    "confidence": classification["confidence"],
                    "analysis": analysis
                }
            
            elif classification["format"] == "EMAIL":
                email_prompt = self._create_email_prompt(doc.page_content)
                email_response = self._call_api(email_prompt)
                analysis = self._parse_json_response(email_response)
                return {
                    "format": classification["format"],
                    "intent": classification["intent"],
                    "confidence": classification["confidence"],
                    "analysis": analysis
                }
            
            elif classification["format"] == "TEXT":
                text_prompt = self._create_text_prompt(doc.page_content)
                text_response = self._call_api(text_prompt)
                analysis = self._parse_json_response(text_response)
                return {
                    "format": classification["format"],
                    "intent": classification["intent"],
                    "confidence": classification["confidence"],
                    "analysis": analysis
                }
            
            elif classification["format"] == "PDF":
                logging.info("Processing PDF document...")
                logging.debug(f"PDF content length: {len(doc.page_content)}")
                pdf_prompt = self._create_pdf_prompt(doc.page_content)
                pdf_response = self._call_api(pdf_prompt)
                logging.debug(f"PDF API response: {pdf_response}")
                analysis = self._parse_json_response(pdf_response)
                logging.info("PDF processing completed successfully")
                return {
                    "format": classification["format"],
                    "intent": classification["intent"],
                    "confidence": classification["confidence"],
                    "analysis": analysis
                }
            
            else:
                logging.warning(f"Unsupported format: {classification['format']}")
                return {
                    "format": classification["format"],
                    "intent": classification["intent"],
                    "confidence": classification["confidence"],
                    "analysis": {"error": "Unsupported format"}
                }
                
        except Exception as e:
            logging.error(f"Error processing document: {str(e)}")
            return {
                "format": "UNKNOWN",
                "intent": "UNKNOWN",
                "confidence": 0.0,
                "analysis": {"error": str(e)}
            }