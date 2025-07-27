import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class BertAutoencoder(nn.Module):
    """
    Simple autoencoder for BERT embeddings.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 64):
        super(BertAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

class BertAnomalyDetector:
    """
    BERT-based anomaly detection for time-series sequences.
    Uses pre-trained BERT to extract features and autoencoder for anomaly detection.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 512,
                 device: Optional[str] = None):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize BERT components
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert_model = BertModel.from_pretrained(model_name)
        self.bert_model.to(self.device)
        self.bert_model.eval()
        
        # Autoencoder components
        self.autoencoder = None
        self.scaler = StandardScaler()
        self.isolation_forest = None
        self.is_fitted = False
        
        # Training parameters
        self.autoencoder_hidden_dim = 128
        self.autoencoder_latent_dim = 64
        self.learning_rate = 1e-3
        self.num_epochs = 10
        self.batch_size = 32
        
        print(f"BERT Anomaly Detector initialized on {self.device}")
    
    def extract_bert_features(self, sequences: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Extract BERT features from text sequences.
        
        Args:
            sequences: Dictionary mapping user IDs to text sequences
            
        Returns:
            Dictionary mapping user IDs to BERT embeddings
        """
        print("Extracting BERT features...")
        
        user_embeddings = {}
        
        with torch.no_grad():
            for user, sequence in sequences.items():
                try:
                    # Tokenize sequence
                    inputs = self.tokenizer(
                        sequence,
                        truncation=True,
                        padding=True,
                        max_length=self.max_length,
                        return_tensors="pt"
                    )
                    
                    # Move to device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Get BERT outputs
                    outputs = self.bert_model(**inputs)
                    
                    # Use pooled output ([CLS] token)
                    pooled_output = outputs.pooler_output
                    
                    # Convert to numpy
                    embedding = pooled_output.cpu().numpy().flatten()
                    user_embeddings[user] = embedding
                    
                except Exception as e:
                    print(f"Error processing user {user}: {e}")
                    continue
        
        return user_embeddings
    
    def fit(self, user_embeddings: Dict[str, np.ndarray], 
            normal_users: Optional[List[str]] = None) -> None:
        """
        Fit the BERT autoencoder on user embeddings.
        
        Args:
            user_embeddings: Dictionary mapping user IDs to their BERT embeddings
            normal_users: List of user IDs considered "normal" for training
        """
        print("Fitting BERT autoencoder...")
        
        if normal_users is None:
            normal_users = list(user_embeddings.keys())
        
        # Collect embeddings from normal users
        normal_embeddings = []
        for user in normal_users:
            if user in user_embeddings and len(user_embeddings[user]) > 0:
                normal_embeddings.append(user_embeddings[user])
        
        if not normal_embeddings:
            raise ValueError("No valid embeddings found for training")
        
        normal_embeddings = np.array(normal_embeddings)
        
        # Scale embeddings
        normal_embeddings_scaled = self.scaler.fit_transform(normal_embeddings)
        
        # Initialize autoencoder
        input_dim = normal_embeddings_scaled.shape[1]
        self.autoencoder = BertAutoencoder(
            input_dim=input_dim,
            hidden_dim=self.autoencoder_hidden_dim,
            latent_dim=self.autoencoder_latent_dim
        ).to(self.device)
        
        # Train autoencoder
        self._train_autoencoder(normal_embeddings_scaled)
        
        # Fit isolation forest on reconstruction errors
        self._fit_isolation_forest(normal_users, user_embeddings)
        
        self.is_fitted = True
        print("BERT autoencoder fitted successfully")
    
    def _train_autoencoder(self, embeddings: np.ndarray) -> None:
        """Train the autoencoder on normal embeddings."""
        print("Training autoencoder...")
        
        # Convert to PyTorch tensors
        embeddings_tensor = torch.FloatTensor(embeddings).to(self.device)
        
        # Initialize optimizer and loss function
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        self.autoencoder.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            num_batches = 0
            
            # Process in batches
            for i in range(0, len(embeddings_tensor), self.batch_size):
                batch = embeddings_tensor[i:i + self.batch_size]
                
                # Forward pass
                reconstructed = self.autoencoder(batch)
                loss = criterion(reconstructed, batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if (epoch + 1) % 5 == 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.6f}")
    
    def _fit_isolation_forest(self, normal_users: List[str], 
                             user_embeddings: Dict[str, np.ndarray]) -> None:
        """Fit isolation forest on reconstruction errors."""
        print("Fitting isolation forest...")
        
        reconstruction_errors = []
        
        self.autoencoder.eval()
        with torch.no_grad():
            for user in normal_users:
                if user in user_embeddings and len(user_embeddings[user]) > 0:
                    # Get embedding
                    embedding = user_embeddings[user]
                    embedding_scaled = self.scaler.transform(embedding.reshape(1, -1))
                    embedding_tensor = torch.FloatTensor(embedding_scaled).to(self.device)
                    
                    # Get reconstruction
                    reconstructed = self.autoencoder(embedding_tensor)
                    
                    # Compute reconstruction error
                    error = torch.mean((embedding_tensor - reconstructed) ** 2).item()
                    reconstruction_errors.append(error)
        
        if reconstruction_errors:
            reconstruction_errors = np.array(reconstruction_errors).reshape(-1, 1)
            self.isolation_forest = IsolationForest(
                contamination=0.1,  # Assume 10% of data is anomalous
                random_state=42
            )
            self.isolation_forest.fit(reconstruction_errors)
    
    def detect_anomalies(self, user_embeddings: Dict[str, np.ndarray],
                        threshold_multiplier: float = 3.0) -> Dict[str, Dict]:
        """
        Detect anomalies in user sequences using BERT autoencoder.
        
        Args:
            user_embeddings: Dictionary mapping user IDs to their BERT embeddings
            threshold_multiplier: Multiplier for anomaly threshold
            
        Returns:
            Dictionary mapping user IDs to anomaly detection results
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        print("Detecting anomalies using BERT model...")
        
        results = {}
        
        self.autoencoder.eval()
        with torch.no_grad():
            for user, embedding in user_embeddings.items():
                try:
                    # Scale embedding
                    embedding_scaled = self.scaler.transform(embedding.reshape(1, -1))
                    embedding_tensor = torch.FloatTensor(embedding_scaled).to(self.device)
                    
                    # Get reconstruction
                    reconstructed = self.autoencoder(embedding_tensor)
                    
                    # Compute reconstruction error
                    reconstruction_error = torch.mean((embedding_tensor - reconstructed) ** 2).item()
                    
                    # Use isolation forest for anomaly detection
                    if self.isolation_forest is not None:
                        anomaly_score = self.isolation_forest.decision_function([[reconstruction_error]])[0]
                        is_anomaly = self.isolation_forest.predict([[reconstruction_error]])[0] == -1
                    else:
                        # Fallback to threshold-based detection
                        # Use reconstruction error distribution from training data
                        errors = []
                        for train_user in user_embeddings.keys():
                            if train_user != user:
                                train_embedding = user_embeddings[train_user]
                                train_embedding_scaled = self.scaler.transform(train_embedding.reshape(1, -1))
                                train_embedding_tensor = torch.FloatTensor(train_embedding_scaled).to(self.device)
                                train_reconstructed = self.autoencoder(train_embedding_tensor)
                                train_error = torch.mean((train_embedding_tensor - train_reconstructed) ** 2).item()
                                errors.append(train_error)
                        
                        if errors:
                            mean_error = np.mean(errors)
                            std_error = np.std(errors)
                            threshold = mean_error + threshold_multiplier * std_error
                            is_anomaly = reconstruction_error > threshold
                            anomaly_score = (reconstruction_error - mean_error) / (std_error + 1e-8)
                        else:
                            is_anomaly = False
                            anomaly_score = 0.0
                    
                    # Additional analysis
                    embedding_analysis = self._analyze_embedding(embedding)
                    
                    results[user] = {
                        'is_anomaly': bool(is_anomaly),
                        'anomaly_score': float(anomaly_score),
                        'reconstruction_error': float(reconstruction_error),
                        'embedding_norm': float(np.linalg.norm(embedding)),
                        'embedding_analysis': embedding_analysis,
                        'explanation': self._generate_explanation(
                            is_anomaly, reconstruction_error, embedding_analysis
                        )
                    }
                    
                except Exception as e:
                    print(f"Error processing user {user}: {e}")
                    continue
        
        return results
    
    def _analyze_embedding(self, embedding: np.ndarray) -> Dict:
        """Analyze BERT embedding characteristics."""
        return {
            'norm': float(np.linalg.norm(embedding)),
            'mean': float(np.mean(embedding)),
            'std': float(np.std(embedding)),
            'max_value': float(np.max(embedding)),
            'min_value': float(np.min(embedding)),
            'sparsity': float(np.sum(np.abs(embedding) < 1e-6) / len(embedding))
        }
    
    def _generate_explanation(self, is_anomaly: bool, reconstruction_error: float,
                            embedding_analysis: Dict) -> str:
        """Generate human-readable explanation of anomaly detection results."""
        if is_anomaly:
            explanation = f"Anomaly detected with high reconstruction error ({reconstruction_error:.4f}). "
            explanation += f"Embedding norm: {embedding_analysis['norm']:.3f}, "
            explanation += f"Embedding sparsity: {embedding_analysis['sparsity']:.3f}. "
            
            if reconstruction_error > 0.1:
                explanation += "Very high reconstruction error indicates unusual sequence patterns."
            elif reconstruction_error > 0.05:
                explanation += "High reconstruction error indicates moderately unusual patterns."
            else:
                explanation += "Moderate reconstruction error indicates slightly unusual patterns."
        else:
            explanation = f"Normal sequence with reconstruction error {reconstruction_error:.4f}. "
            explanation += f"Embedding norm: {embedding_analysis['norm']:.3f}, "
            explanation += f"Embedding sparsity: {embedding_analysis['sparsity']:.3f}. "
            explanation += "Reconstruction error is within normal range."
        
        return explanation
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of BERT embeddings."""
        return self.bert_model.config.hidden_size
    
    def encode_sequence(self, sequence: str) -> np.ndarray:
        """Encode a single sequence to BERT embedding."""
        with torch.no_grad():
            inputs = self.tokenizer(
                sequence,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.bert_model(**inputs)
            embedding = outputs.pooler_output.cpu().numpy().flatten()
            
            return embedding 