import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class CERTDataProcessor:
    """
    Data processor for CERT Insider Threat dataset.
    Handles loading, merging, and feature engineering for time-series analysis.
    """
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.merged_data = None
        self.user_sequences = {}
        self.user_features = {}
        
    def load_sample_data(self) -> pd.DataFrame:
        """
        Load sample CSV files from the CERT dataset.
        Creates synthetic data if actual files are not available.
        """
        print("Loading CERT Insider Threat dataset...")
        
        # Check if actual data files exist
        logon_file = os.path.join(self.data_dir, "logon.csv")
        device_file = os.path.join(self.data_dir, "device.csv")
        
        if os.path.exists(logon_file) and os.path.exists(device_file):
            # Load actual data
            logon_data = pd.read_csv(logon_file)
            device_data = pd.read_csv(device_file)
        else:
            # Create synthetic data for demonstration
            print("Actual data files not found. Creating synthetic data...")
            logon_data, device_data = self._create_synthetic_data()
        
        # Merge datasets
        self.merged_data = self._merge_datasets(logon_data, device_data)
        return self.merged_data
    
    def _create_synthetic_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create synthetic CERT-like data for demonstration."""
        np.random.seed(42)
        
        # Generate synthetic logon data
        n_records = 10000
        users = [f"USER{i:04d}" for i in range(1, 101)]
        computers = [f"PC{i:03d}" for i in range(1, 51)]
        
        # Generate timestamps over 6 months
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 6, 30)
        timestamps = []
        
        for _ in range(n_records):
            timestamp = start_date + timedelta(
                days=np.random.randint(0, (end_date - start_date).days),
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60)
            )
            timestamps.append(timestamp)
        
        logon_data = pd.DataFrame({
            'date': timestamps,
            'user': np.random.choice(users, n_records),
            'computer': np.random.choice(computers, n_records),
            'activity': np.random.choice(['logon', 'logoff'], n_records, p=[0.6, 0.4])
        })
        
        # Generate synthetic device data
        device_records = int(n_records * 0.3)
        device_timestamps = []
        
        for _ in range(device_records):
            timestamp = start_date + timedelta(
                days=np.random.randint(0, (end_date - start_date).days),
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60)
            )
            device_timestamps.append(timestamp)
        
        device_data = pd.DataFrame({
            'date': device_timestamps,
            'user': np.random.choice(users, device_records),
            'computer': np.random.choice(computers, device_records),
            'activity': np.random.choice(['connect', 'disconnect'], device_records, p=[0.7, 0.3])
        })
        
        return logon_data, device_data
    
    def _merge_datasets(self, logon_data: pd.DataFrame, device_data: pd.DataFrame) -> pd.DataFrame:
        """Merge logon and device datasets."""
        # Ensure date column is datetime
        logon_data['date'] = pd.to_datetime(logon_data['date'])
        device_data['date'] = pd.to_datetime(device_data['date'])
        
        # Add source identifier
        logon_data['source'] = 'logon'
        device_data['source'] = 'device'
        
        # Merge datasets
        merged = pd.concat([logon_data, device_data], ignore_index=True)
        
        # Sort by user and timestamp
        merged = merged.sort_values(['user', 'date'])
        
        return merged
    
    def engineer_features(self) -> Dict[str, pd.DataFrame]:
        """
        Perform time-series feature engineering.
        Groups by user and daily windows, extracts features.
        """
        print("Engineering time-series features...")
        
        if self.merged_data is None:
            raise ValueError("Data not loaded. Call load_sample_data() first.")
        
        # Convert date to datetime if not already
        self.merged_data['date'] = pd.to_datetime(self.merged_data['date'])
        
        # Add date features
        self.merged_data['date_only'] = self.merged_data['date'].dt.date
        self.merged_data['hour'] = self.merged_data['date'].dt.hour
        self.merged_data['day_of_week'] = self.merged_data['date'].dt.dayofweek
        
        # Group by user and daily windows
        daily_features = []
        
        for user in tqdm(self.merged_data['user'].unique(), desc="Processing users"):
            user_data = self.merged_data[self.merged_data['user'] == user].copy()
            
            # Group by date
            daily_groups = user_data.groupby('date_only')
            
            for date, group in daily_groups:
                features = {
                    'user': user,
                    'date': date,
                    'total_activities': len(group),
                    'logon_count': len(group[group['activity'] == 'logon']),
                    'logoff_count': len(group[group['activity'] == 'logoff']),
                    'connect_count': len(group[group['activity'] == 'connect']),
                    'disconnect_count': len(group[group['activity'] == 'disconnect']),
                    'unique_computers': group['computer'].nunique(),
                    'avg_hour': group['hour'].mean(),
                    'std_hour': group['hour'].std(),
                    'early_morning_activity': len(group[group['hour'] < 6]),
                    'late_night_activity': len(group[group['hour'] > 22]),
                    'weekend_activity': len(group[group['day_of_week'] >= 5])
                }
                daily_features.append(features)
        
        # Create features DataFrame
        features_df = pd.DataFrame(daily_features)
        
        # Add rolling statistics
        for user in features_df['user'].unique():
            user_mask = features_df['user'] == user
            user_data = features_df[user_mask].sort_values('date')
            
            # Rolling mean and std for key features
            for col in ['total_activities', 'logon_count', 'unique_computers']:
                features_df.loc[user_mask, f'{col}_rolling_mean'] = user_data[col].rolling(window=7, min_periods=1).mean().values
                features_df.loc[user_mask, f'{col}_rolling_std'] = user_data[col].rolling(window=7, min_periods=1).std().values
        
        # Calculate anomaly scores using simple statistical methods
        for col in ['total_activities', 'logon_count', 'unique_computers']:
            rolling_mean_col = f'{col}_rolling_mean'
            rolling_std_col = f'{col}_rolling_std'
            
            # Z-score based anomaly score
            features_df[f'{col}_anomaly_score'] = np.abs(
                (features_df[col] - features_df[rolling_mean_col]) / 
                (features_df[rolling_std_col] + 1e-8)
            )
        
        self.user_features = features_df
        return {'daily_features': features_df}
    
    def textualize_events(self) -> Dict[str, List[str]]:
        """
        Convert events to textual format for embeddings.
        Returns per-user sequences of textualized events.
        """
        print("Textualizing events...")
        
        if self.merged_data is None:
            raise ValueError("Data not loaded. Call load_sample_data() first.")
        
        user_sequences = {}
        
        for user in tqdm(self.merged_data['user'].unique(), desc="Textualizing events"):
            user_data = self.merged_data[self.merged_data['user'] == user].sort_values('date')
            
            textualized_events = []
            for _, row in user_data.iterrows():
                timestamp = row['date'].strftime('%Y-%m-%d %H:%M')
                event_text = f"User {row['user']} {row['activity']} at {timestamp} via {row['computer']}"
                textualized_events.append(event_text)
            
            user_sequences[user] = textualized_events
        
        self.user_sequences = user_sequences
        return user_sequences
    
    def prepare_bert_sequences(self, max_length: int = 512) -> Dict[str, str]:
        """
        Prepare concatenated text sequences for BERT input.
        Returns per-user concatenated sequences.
        """
        print("Preparing BERT sequences...")
        
        if not self.user_sequences:
            self.textualize_events()
        
        bert_sequences = {}
        
        for user, events in self.user_sequences.items():
            # Concatenate events with separator
            concatenated = " [SEP] ".join(events)
            
            # Truncate if too long
            if len(concatenated) > max_length:
                concatenated = concatenated[:max_length]
            
            bert_sequences[user] = concatenated
        
        return bert_sequences
    
    def get_user_activity_summary(self, user_id: str) -> Dict:
        """Get summary statistics for a specific user."""
        if self.user_features is None:
            self.engineer_features()
        
        user_data = self.user_features[self.user_features['user'] == user_id]
        
        if user_data.empty:
            return {"error": f"User {user_id} not found"}
        
        summary = {
            'user_id': user_id,
            'total_days': len(user_data),
            'avg_daily_activities': user_data['total_activities'].mean(),
            'max_daily_activities': user_data['total_activities'].max(),
            'avg_logons_per_day': user_data['logon_count'].mean(),
            'avg_computers_per_day': user_data['unique_computers'].mean(),
            'weekend_activity_ratio': user_data['weekend_activity'].sum() / len(user_data),
            'late_night_activity_ratio': user_data['late_night_activity'].sum() / len(user_data),
            'recent_anomaly_scores': user_data[['total_activities_anomaly_score', 
                                               'logon_count_anomaly_score', 
                                               'unique_computers_anomaly_score']].tail(5).to_dict('records')
        }
        
        return summary 