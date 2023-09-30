import torch
import numpy as np
from sentence_transformers import SentenceTransformer

def embedding_texts(texts: list[str], model_name: str) -> np.ndarray:
    """
    Embedding texts using SentenceTransformer model

    Args:
        texts (list[str]): list of texts
        model_name (str): model name
    
    Returns:
        np.ndarray: embeddings
    """

    model = SentenceTransformer(model_name)
    embedding_texts =  model.encode(
        texts, 
        show_progress_bar=True, 
        device="cuda" if torch.cuda.is_available() else "cpu", 
        normalize_embeddings=True,
    )

    # delete model to free memory
    del model
    return embedding_texts