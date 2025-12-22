"""
Utility functions for text processing using local LLM.
"""
import requests
import json
import logging
from typing import Optional, Dict, Any

# Configuration
LLM_API_URL = "http://llm:11435/api/chat"  # Update with your local LLM API endpoint
HEADERS = {
    "Content-Type": "application/json",
}

def filter_profanity(text: str) -> str:
    """
    Send text to local LLM for profanity filtering.
    
    Args:
        text: Input text to filter
        
    Returns:
        Filtered text with profanity replaced or removed
    """
    if not text or not isinstance(text, str):
        return text
        
    try:
        # Prepare the prompt for the LLM
        prompt = f"""
        Замени нецензурные и оскорбительные слова на _____. 
        Сохрани все остальное без изменений, включая пунктуацию и форматирование.
        
        Входной текст: {"{text}"}
        
        Отфильтрованный текст: """
        
        payload = {
            "model": "Qwen3-30B-A3B-Instruct-2507:latest",
            "messages":[
                {
                    "role":"system",
                    "content":"Ты - помощник для фильтрации нецензурной лексики и форматированию текстка. Заменяй только \n нецензурные слова на _____, отформатируй текст по пунктуации и убери повторяющиеся подряд слова, можно \n менять неправильные термины (когда четко понятно что это некорректно). Учти что в коллективе могут быть \n люди с дифектами речи (заикание и тд). Словарь для подбора текстов должен быть профессиональный: БАНК, \n IT, Бизнес. Выводишь в результат только текст, никаких свих размышлений \n Пример дефектов и их замены \n (диффект - замена): \n Нет нет нет - нет \n Нет, нет - нет \n Дополнительный словарь: \n Бэклог, DEV, \n видяйка это VDI, UnixSAN"
                },
                {
                    "role":"user",
                    "content": text
                }
            ],
            
            "stream": False
        }
        
        response = requests.post(
            LLM_API_URL,
            headers=HEADERS,
            json=payload,
            timeout=60  # 10 seconds timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            filtered_text = result['message'][0]['content'].strip('"')
            
            # Log if text was modified
            if filtered_text != text:
                logging.info("Text was filtered for profanity")
                
            return filtered_text
        else:
            logging.error(f"Error calling LLM API: {response.status_code} - {response.text}")
            return text
            
    except Exception as e:
        logging.error(f"Error in LLM profanity filter: {e}")
        return text

def clean_segment_text(segment: dict, text_field: str = 'text') -> dict:
    """
    Clean text in a segment dictionary using LLM.
    
    Args:
        segment: Dictionary containing text to clean
        text_field: Key of the text field in the dictionary
        
    Returns:
        Dictionary with cleaned text
    """
    if not isinstance(segment, dict) or text_field not in segment:
        return segment
        
    cleaned_text = filter_profanity(segment[text_field])
    return {**segment, text_field: cleaned_text}
