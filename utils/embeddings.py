
import numpy as np
import pandas as pd
import sqlite3
from sentence_transformers import SentenceTransformer
import config

class EmbeddingEngine:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.ids_df = None
        self.load_data()

    def load_data(self):
        try:
            self.embeddings = np.load(config.EMBEDDINGS_PATH)
            self.ids_df = pd.read_csv(config.EMBEDDINGS_IDS_PATH)
            print(f"Loaded embeddings: {self.embeddings.shape}")
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            # Initialize empty if files don't exist yet
            self.embeddings = np.array([])
            self.ids_df = pd.DataFrame()

    def search(self, query, top_k=5):
        if self.embeddings.size == 0:
            return []

        # Encode query
        query_emb = self.model.encode([query])
        
        # Calculate cosine similarity
        similarities = np.dot(self.embeddings, query_emb.T).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Retrieve IDs
        top_ids = self.ids_df.iloc[top_indices]['id'].tolist()
        
        return top_ids
