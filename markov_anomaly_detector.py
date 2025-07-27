import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MarkovAnomalyDetector:
    """
    Markov Chain-based anomaly detection for time-series sequences.
    Uses K-means clustering to discretize embeddings into states,
    builds transition probability matrices, and detects anomalies.
    """
    
    def __init__(self, n_clusters: int = 10, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.transition_matrices = {}
        self.scaler = StandardScaler()
        self.isolation_forest = None
        self.is_fitted = False
        
    def fit(self, user_embeddings: Dict[str, np.ndarray], 
            normal_users: Optional[List[str]] = None) -> None:
        """
        Fit the Markov Chain model on user embeddings.
        
        Args:
            user_embeddings: Dictionary mapping user IDs to their embeddings
            normal_users: List of user IDs considered "normal" for training
        """
        print("Fitting Markov Chain anomaly detector...")
        
        if normal_users is None:
            # Use all users for training if no normal users specified
            normal_users = list(user_embeddings.keys())
        
        # Collect embeddings from normal users
        normal_embeddings = []
        for user in normal_users:
            if user in user_embeddings and len(user_embeddings[user]) > 0:
                normal_embeddings.extend(user_embeddings[user])
        
        if not normal_embeddings:
            raise ValueError("No valid embeddings found for training")
        
        normal_embeddings = np.array(normal_embeddings)
        
        # Scale embeddings
        normal_embeddings_scaled = self.scaler.fit_transform(normal_embeddings)
        
        # Fit K-means clustering to discretize embeddings into states
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.kmeans.fit(normal_embeddings_scaled)
        
        # Build transition probability matrices for each normal user
        for user in normal_users:
            if user in user_embeddings and len(user_embeddings[user]) > 1:
                user_embeddings_scaled = self.scaler.transform(user_embeddings[user])
                states = self.kmeans.predict(user_embeddings_scaled)
                transition_matrix = self._build_transition_matrix(states)
                self.transition_matrices[user] = transition_matrix
        
        # Fit Isolation Forest for outlier detection
        self.isolation_forest = IsolationForest(
            contamination=0.1,  # Assume 10% of data is anomalous
            random_state=self.random_state
        )
        
        # Use sequence likelihoods as features for isolation forest
        likelihoods = []
        for user in normal_users:
            if user in user_embeddings and len(user_embeddings[user]) > 1:
                user_embeddings_scaled = self.scaler.transform(user_embeddings[user])
                states = self.kmeans.predict(user_embeddings_scaled)
                likelihood = self._compute_sequence_likelihood(states, user)
                likelihoods.append(likelihood)
        
        if likelihoods:
            likelihoods = np.array(likelihoods).reshape(-1, 1)
            self.isolation_forest.fit(likelihoods)
        
        self.is_fitted = True
        print(f"Markov Chain model fitted with {len(self.transition_matrices)} users")
    
    def _build_transition_matrix(self, states: np.ndarray) -> np.ndarray:
        """
        Build transition probability matrix from state sequence.
        
        Args:
            states: Array of state labels
            
        Returns:
            Transition probability matrix
        """
        n_states = self.n_clusters
        transition_matrix = np.zeros((n_states, n_states))
        
        # Count transitions
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            transition_matrix[current_state, next_state] += 1
        
        # Normalize to get probabilities
        row_sums = transition_matrix.sum(axis=1)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        transition_matrix = transition_matrix / row_sums[:, np.newaxis]
        
        return transition_matrix
    
    def _compute_sequence_likelihood(self, states: np.ndarray, user: str) -> float:
        """
        Compute likelihood of a state sequence using the user's transition matrix.
        
        Args:
            states: Array of state labels
            user: User ID for transition matrix lookup
            
        Returns:
            Log likelihood of the sequence
        """
        if user not in self.transition_matrices:
            return 0.0
        
        transition_matrix = self.transition_matrices[user]
        log_likelihood = 0.0
        
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            
            # Get transition probability
            prob = transition_matrix[current_state, next_state]
            
            # Avoid log(0)
            if prob > 0:
                log_likelihood += np.log(prob)
            else:
                log_likelihood += np.log(1e-10)  # Small probability for zero transitions
        
        return log_likelihood
    
    def detect_anomalies(self, user_embeddings: Dict[str, np.ndarray], 
                        threshold_multiplier: float = 3.0) -> Dict[str, Dict]:
        """
        Detect anomalies in user sequences.
        
        Args:
            user_embeddings: Dictionary mapping user IDs to their embeddings
            threshold_multiplier: Multiplier for anomaly threshold
            
        Returns:
            Dictionary mapping user IDs to anomaly detection results
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        print("Detecting anomalies using Markov Chain model...")
        
        results = {}
        
        for user, embeddings in user_embeddings.items():
            if len(embeddings) < 2:
                continue
            
            # Scale embeddings
            embeddings_scaled = self.scaler.transform(embeddings)
            
            # Predict states
            states = self.kmeans.predict(embeddings_scaled)
            
            # Compute sequence likelihood
            likelihood = self._compute_sequence_likelihood(states, user)
            
            # Use isolation forest for outlier detection
            if self.isolation_forest is not None:
                anomaly_score = self.isolation_forest.decision_function([[likelihood]])[0]
                is_anomaly = self.isolation_forest.predict([[likelihood]])[0] == -1
            else:
                # Fallback to threshold-based detection
                # Use likelihood distribution from training data
                likelihoods = []
                for train_user in self.transition_matrices.keys():
                    if train_user in user_embeddings and len(user_embeddings[train_user]) > 1:
                        train_embeddings_scaled = self.scaler.transform(user_embeddings[train_user])
                        train_states = self.kmeans.predict(train_embeddings_scaled)
                        train_likelihood = self._compute_sequence_likelihood(train_states, train_user)
                        likelihoods.append(train_likelihood)
                
                if likelihoods:
                    mean_likelihood = np.mean(likelihoods)
                    std_likelihood = np.std(likelihoods)
                    threshold = mean_likelihood - threshold_multiplier * std_likelihood
                    is_anomaly = likelihood < threshold
                    anomaly_score = (likelihood - mean_likelihood) / (std_likelihood + 1e-8)
                else:
                    is_anomaly = False
                    anomaly_score = 0.0
            
            # Additional analysis
            state_transitions = self._analyze_state_transitions(states)
            
            results[user] = {
                'is_anomaly': bool(is_anomaly),
                'anomaly_score': float(anomaly_score),
                'sequence_likelihood': float(likelihood),
                'sequence_length': len(states),
                'unique_states': len(np.unique(states)),
                'state_transitions': state_transitions,
                'explanation': self._generate_explanation(
                    is_anomaly, likelihood, len(states), state_transitions
                )
            }
        
        return results
    
    def _analyze_state_transitions(self, states: np.ndarray) -> Dict:
        """Analyze state transition patterns."""
        if len(states) < 2:
            return {}
        
        transitions = {}
        for i in range(len(states) - 1):
            transition = f"{states[i]}->{states[i+1]}"
            transitions[transition] = transitions.get(transition, 0) + 1
        
        return {
            'total_transitions': len(states) - 1,
            'unique_transitions': len(transitions),
            'most_common_transition': max(transitions.items(), key=lambda x: x[1]) if transitions else None,
            'transition_entropy': self._compute_entropy(list(transitions.values()))
        }
    
    def _compute_entropy(self, values: List[int]) -> float:
        """Compute entropy of transition distribution."""
        if not values:
            return 0.0
        
        total = sum(values)
        if total == 0:
            return 0.0
        
        probabilities = [v / total for v in values]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        return entropy
    
    def _generate_explanation(self, is_anomaly: bool, likelihood: float, 
                            sequence_length: int, transitions: Dict) -> str:
        """Generate human-readable explanation of anomaly detection results."""
        if is_anomaly:
            explanation = f"Anomaly detected in sequence with low likelihood ({likelihood:.4f}). "
            explanation += f"Sequence length: {sequence_length}, "
            explanation += f"Unique transitions: {transitions.get('unique_transitions', 0)}. "
            
            if likelihood < -10:
                explanation += "Very unusual transition patterns detected."
            elif likelihood < -5:
                explanation += "Moderately unusual transition patterns detected."
            else:
                explanation += "Slightly unusual transition patterns detected."
        else:
            explanation = f"Normal sequence with likelihood {likelihood:.4f}. "
            explanation += f"Sequence length: {sequence_length}, "
            explanation += f"Unique transitions: {transitions.get('unique_transitions', 0)}. "
            explanation += "Transition patterns appear normal."
        
        return explanation
    
    def get_user_transition_matrix(self, user: str) -> Optional[np.ndarray]:
        """Get transition probability matrix for a specific user."""
        return self.transition_matrices.get(user)
    
    def get_state_embeddings(self) -> np.ndarray:
        """Get cluster centers (state embeddings) from K-means."""
        if self.kmeans is None:
            return np.array([])
        return self.kmeans.cluster_centers_
    
    def predict_states(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict states for given embeddings."""
        if self.kmeans is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        embeddings_scaled = self.scaler.transform(embeddings)
        return self.kmeans.predict(embeddings_scaled) 