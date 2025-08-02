import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple, Optional
import pickle
import os
from tqdm import tqdm

class VectorStore:
    """
    Vector store for storing and retrieving embeddings of time-series sequences.
    Uses FAISS for efficient similarity search.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []
        self.embeddings = []
        
    def create_embeddings(self, texts: List[str], metadata: List[Dict]) -> np.ndarray:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            metadata: List of metadata dictionaries corresponding to texts
            
        Returns:
            numpy array of embeddings
        """
        print(f"Creating embeddings using {self.model_name}...")
        
        # Create embeddings
        embeddings = self.encoder.encode(texts, show_progress_bar=True, batch_size=32)
        
        # Store metadata
        self.metadata.extend(metadata)
        self.embeddings.extend(embeddings)
        
        return embeddings
    
    def build_index(self, embeddings: np.ndarray, index_type: str = "l2") -> None:
        """
        Build FAISS index for similarity search.
        
        Args:
            embeddings: numpy array of embeddings
            index_type: type of index ("l2", "ip", "cosine")
        """
        print("Building FAISS index...")
        
        dimension = embeddings.shape[1]
        
        if index_type == "l2":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "ip":
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "cosine":
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index = faiss.IndexFlatIP(dimension)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Add vectors to index
        self.index.add(embeddings.astype('float32'))
        
        print(f"Index built with {self.index.ntotal} vectors")
    
    def search_similar(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for similar sequences.
        
        Args:
            query: Query text
            k: Number of similar sequences to return
            
        Returns:
            List of dictionaries with similarity scores and metadata
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.encoder.encode([query])
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                result = {
                    'rank': i + 1,
                    'similarity_score': 1.0 / (1.0 + distance),  # Convert distance to similarity
                    'distance': float(distance),
                    'metadata': self.metadata[idx]
                }
                results.append(result)
        
        return results
    
    def search_by_user(self, user_id: str, k: int = 10) -> List[Dict]:
        """
        Search for sequences from a specific user.
        
        Args:
            user_id: User ID to search for
            k: Number of results to return
            
        Returns:
            List of dictionaries with user sequences
        """
        user_sequences = []
        
        for i, metadata in enumerate(self.metadata):
            if metadata.get('user') == user_id:
                user_sequences.append({
                    'index': i,
                    'metadata': metadata,
                    'embedding': self.embeddings[i] if i < len(self.embeddings) else None
                })
        
        # Sort by timestamp if available
        user_sequences.sort(key=lambda x: x['metadata'].get('timestamp', 0))
        
        return user_sequences[:k]
    
    def get_user_embeddings(self, user_id: str) -> np.ndarray:
        """
        Get all embeddings for a specific user.
        
        Args:
            user_id: User ID
            
        Returns:
            numpy array of embeddings for the user
        """
        user_embeddings = []
        
        for i, metadata in enumerate(self.metadata):
            if metadata.get('user') == user_id and i < len(self.embeddings):
                user_embeddings.append(self.embeddings[i])
        
        return np.array(user_embeddings) if user_embeddings else np.array([])
    
    def save_index(self, filepath: str) -> None:
        """Save the index and metadata to disk."""
        print(f"Saving index to {filepath}...")
        
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # Save metadata and embeddings
        with open(f"{filepath}_metadata.pkl", 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'embeddings': self.embeddings,
                'model_name': self.model_name
            }, f)
    
    def load_index(self, filepath: str) -> None:
        """Load the index and metadata from disk."""
        print(f"Loading index from {filepath}...")
        
        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.faiss")
        
        # Load metadata and embeddings
        with open(f"{filepath}_metadata.pkl", 'rb') as f:
            data = pickle.load(f)
            self.metadata = data['metadata']
            self.embeddings = data['embeddings']
            self.model_name = data['model_name']
        
        # Reload encoder
        self.encoder = SentenceTransformer(self.model_name)
        
        print(f"Loaded index with {self.index.ntotal} vectors")

class TimeSeriesVectorStore:
    """
    Specialized vector store for time-series data with additional functionality.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.vector_store = VectorStore(model_name)
        self.feature_embeddings = {}
        
    def create_feature_embeddings(self, features_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Create embeddings for time-series features.
        
        Args:
            features_df: DataFrame with time-series features
            
        Returns:
            Dictionary mapping user IDs to their feature embeddings
        """
        print("Creating feature embeddings...")
        
        user_embeddings = {}
        
        for user in tqdm(features_df['user'].unique(), desc="Processing users"):
            user_data = features_df[features_df['user'] == user].sort_values('date')
            
            # Create feature text representation
            feature_texts = []
            for _, row in user_data.iterrows():
                feature_text = (
                    f"User {user} on {row['date']}: "
                    f"{row['total_activities']} activities, "
                    f"{row['logon_count']} logons, "
                    f"{row['unique_computers']} computers, "
                    f"anomaly scores: {row.get('total_activities_anomaly_score', 0):.3f}, "
                    f"{row.get('logon_count_anomaly_score', 0):.3f}, "
                    f"{row.get('unique_computers_anomaly_score', 0):.3f}"
                )
                feature_texts.append(feature_text)
            
            # Create embeddings for user's feature sequence
            if feature_texts:
                embeddings = self.vector_store.encoder.encode(feature_texts)
                user_embeddings[user] = embeddings
        
        self.feature_embeddings = user_embeddings
        return user_embeddings
    
    def create_sequence_embeddings(self, user_sequences: Dict[str, List[str]]) -> None:
        """
        Create embeddings for user event sequences and build index.
        
        Args:
            user_sequences: Dictionary mapping user IDs to lists of event texts
        """
        print("Creating sequence embeddings...")
        
        texts = []
        metadata = []
        
        for user, events in user_sequences.items():
            # Create sequence text
            sequence_text = " [SEP] ".join(events)
            
            texts.append(sequence_text)
            metadata.append({
                'user': user,
                'sequence_length': len(events),
                'type': 'event_sequence'
            })
        
        # Create embeddings
        embeddings = self.vector_store.create_embeddings(texts, metadata)
        
        # Build index
        self.vector_store.build_index(embeddings)
    
    def search_similar_sequences(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar event sequences."""
        return self.vector_store.search_similar(query, k)
    
    def get_user_sequence_embeddings(self, user_id: str) -> np.ndarray:
        """Get embeddings for a user's event sequences."""
        return self.vector_store.get_user_embeddings(user_id)
    
    def save(self, filepath: str) -> None:
        """Save the vector store."""
        self.vector_store.save_index(filepath)
        
        # Save feature embeddings separately
        with open(f"{filepath}_features.pkl", 'wb') as f:
            pickle.dump(self.feature_embeddings, f)
    
    def load(self, filepath: str) -> None:
        """Load the vector store."""
        self.vector_store.load_index(filepath)
        
        # Load feature embeddings
        try:
            with open(f"{filepath}_features.pkl", 'rb') as f:
                self.feature_embeddings = pickle.load(f)
        except FileNotFoundError:
            print("Feature embeddings file not found, skipping...") 