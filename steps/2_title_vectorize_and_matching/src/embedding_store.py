import json
import faiss
import torch
import logging
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EmbeddingStore:
    """
    Embedding Store class
    """
    def __init__(self, 
                 model_id: str, 
                 index_path: str, 
                 id_mapping_path: str, 
                 dim:int =768) -> None:
        """
        Embedding Store constructor

        Args:
            model_id (str): model id
            index_path (str): index path
            id_mapping_path (str): id mapping path
            dim (int, optional): embedding dimension. Defaults to 768.

        Returns:
            None
        """
        self.id_mappping = None
        self.model_id = model_id
        self.index_path = index_path
        self.id_mapping_path = id_mapping_path
        self.model = SentenceTransformer(self.model_path, dim=dim)
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))

    def process(self,
                df: pd.DataFrame,
                vect_column: str) -> None:
        """
        Process dataframe and add embeddings to index

        Args:
            df (pd.DataFrame): dataframe
            vect_column (str): column name with texts

        Returns:
            None
        """
        # embedding texts
        logging.info(f'Embedding texts with {self.model_id} model')
        embeddings = self.embedding_texts(df[vect_column].tolist(), self.model_id)
        logging.info(f'Embeddings shape: {embeddings.shape}')

        # mapping text ids to embeddings ids
        logging.info('Mapping text ids to embeddings ids')
        self.id_mappping = {i: text_id for i, text_id in enumerate(df['id'].to_list())}

        # add embeddings to index
        logging.info('Adding embeddings to index')
        self.index.add_with_ids(embeddings, np.array(list(self.id_mappping.keys())))

    def save(self) -> None:
        """
        Save index and matcher

        Returns:
            None
        """
        logging.info(f'Saving index to {self.index_path}')
        faiss.write_index(self.index, self.index_path)

        logging.info(f'Saving id mapping to {self.id_mapping_path}')
        with open(self.id_mapping_path, 'w') as f:
            json.dump(self.id_mappping, f)

    def embedding_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embedding texts using SentenceTransformer model

        Args:
            texts (list[str]): list of texts
        
        Returns:
            np.ndarray: embeddings
        """

        embedding_texts = self.model.encode(
            texts, 
            show_progress_bar=True, 
            device="cuda" if torch.cuda.is_available() else "cpu", 
            normalize_embeddings=True,
        )
        return embedding_texts
