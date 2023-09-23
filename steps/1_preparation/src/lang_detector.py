import torch
from transformers import pipeline

#Define pipeline
langdetector = pipeline("text-classification", model="ERCDiDip/langdetect", max_length=512)

#Define function
def lang_detection_func(item) -> dict:
    """"
    Detect language in item

    Args:
        item (str): item to detect language
    
    Returns:
        dict: item and detected language
    """
    return {'item': item, 'lang_detect': langdetector(item)[0]['label']}