#!/usr/bin/env python3
"""
Agentic RAG Agent with Local LLM Support
Uses Gemma-2B instead of OpenAI for local inference
"""

import os
from typing import Dict, List, Any, Optional
from langchain.agents import AgentExecutor, create_openai_tools_agent, initialize_agent
from langchain.tools import BaseTool
from langchain.schema import BaseMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
import json
from datetime import datetime

# Import local LLM
from local_llm import create_local_llm, LocalLLM

class TimeSeriesQueryInput(BaseModel):
    query: str = Field(description="Query about time-series data or user behavior")

class UserAnalysisInput(BaseModel):
    user_id: str = Field(description="User ID to analyze")

class SequenceRetrievalInput(BaseModel):
    query: str = Field(description="Query to find similar sequences")

class TrendAnalysisInput(BaseModel):
    user_id: str = Field(description="User ID for trend analysis")

class TimeSeriesRetrievalTool(BaseTool):
    name: str = "time_series_retrieval"
    description: str = "Retrieve similar time-series sequences based on a query"
    args_schema: type = SequenceRetrievalInput
    
    def __init__(self, vector_store):
        super().__init__()
        self._vector_store = vector_store
    
    def _run(self, query: str) -> str:
        """Retrieve similar time-series sequences."""
        try:
            # Extract user ID if present in query
            import re
            user_match = re.search(r'USER\d+', query)
            if user_match:
                user_id = user_match.group()
                results = self._vector_store.search_similar_sequences(user_id, 5)
            else:
                results = self._vector_store.search_similar_sequences(query, 5)
            
            if not results:
                return "No similar sequences found."
            
            response = f"Found {len(results)} similar sequences:\n\n"
            for i, result in enumerate(results, 1):
                response += f"{i}. Similarity Score: {result['similarity_score']:.3f}\n"
                response += f"   User: {result['metadata'].get('user', 'Unknown')}\n"
                response += f"   Sequence Length: {result['metadata'].get('sequence_length', 'Unknown')}\n"
                response += f"   Distance: {result['distance']:.3f}\n\n"
            
            return response
        except Exception as e:
            return f"Error retrieving sequences: {str(e)}"

class TrendAnalysisTool(BaseTool):
    name: str = "trend_analysis"
    description: str = "Analyze time-series trends using rolling statistics"
    args_schema: type = TrendAnalysisInput
    
    def __init__(self, data_processor):
        super().__init__()
        self._data_processor = data_processor
    
    def _run(self, user_id: str) -> str:
        """Analyze trends for a specific user."""
        try:
            if self._data_processor.user_features is None:
                return "No feature data available. Please run feature engineering first."
            
            # Extract user ID if present in query
            import re
            user_match = re.search(r'USER\d+', user_id)
            if user_match:
                user_id = user_match.group()
            
            user_data = self._data_processor.user_features[
                self._data_processor.user_features['user'] == user_id
            ].sort_values('date')
            
            if user_data.empty:
                return f"No data found for user {user_id}"
            
            # Calculate trends
            trends = {}
            for col in ['total_activities', 'logon_count', 'unique_computers']:
                if col in user_data.columns:
                    # Calculate rolling mean and trend
                    rolling_mean = user_data[col].rolling(window=7, min_periods=1).mean()
                    trend = (rolling_mean.iloc[-1] - rolling_mean.iloc[0]) / len(rolling_mean)
                    trends[col] = {
                        'current_value': float(user_data[col].iloc[-1]),
                        'rolling_mean': float(rolling_mean.iloc[-1]),
                        'trend': float(trend),
                        'trend_direction': 'increasing' if trend > 0 else 'decreasing' if trend < 0 else 'stable'
                    }
            
            # Format response
            response = f"Trend Analysis for User {user_id} (Window: 7 days):\n\n"
            for feature, trend_data in trends.items():
                response += f"{feature.replace('_', ' ').title()}:\n"
                response += f"  Current Value: {trend_data['current_value']:.2f}\n"
                response += f"  Rolling Mean: {trend_data['rolling_mean']:.2f}\n"
                response += f"  Trend: {trend_data['trend']:.3f} ({trend_data['trend_direction']})\n\n"
            
            return response
        except Exception as e:
            return f"Error analyzing trends: {str(e)}"

class MarkovAnomalyDetectionTool(BaseTool):
    name: str = "markov_anomaly_detection"
    description: str = "Detect anomalies using Markov Chain model"
    args_schema: type = UserAnalysisInput
    
    def __init__(self, markov_detector, vector_store):
        super().__init__()
        self._markov_detector = markov_detector
        self._vector_store = vector_store
    
    def _run(self, user_id: str) -> str:
        """Detect anomalies using Markov Chain model."""
        try:
            if not self._markov_detector.is_fitted:
                return "Markov Chain model not fitted. Please train the model first."
            
            # Extract user ID if present in query
            import re
            user_match = re.search(r'USER\d+', user_id)
            if user_match:
                user_id = user_match.group()
            
            # Get user embeddings
            user_embeddings = self._vector_store.get_user_sequence_embeddings(user_id)
            
            if len(user_embeddings) == 0:
                return f"No embeddings found for user {user_id}"
            
            # Detect anomalies
            results = self._markov_detector.detect_anomalies({user_id: user_embeddings})
            
            if user_id not in results:
                return f"No anomaly detection results for user {user_id}"
            
            result = results[user_id]
            
            response = f"Markov Chain Anomaly Detection Results for User {user_id}:\n\n"
            response += f"Anomaly Detected: {'Yes' if result['is_anomaly'] else 'No'}\n"
            response += f"Anomaly Score: {result['anomaly_score']:.4f}\n"
            response += f"Sequence Likelihood: {result['sequence_likelihood']:.4f}\n"
            response += f"Sequence Length: {result['sequence_length']}\n"
            response += f"Unique States: {result['unique_states']}\n\n"
            response += f"Explanation: {result['explanation']}\n\n"
            
            return response
        except Exception as e:
            return f"Error in Markov anomaly detection: {str(e)}"

class BertAnomalyDetectionTool(BaseTool):
    name: str = "bert_anomaly_detection"
    description: str = "Detect anomalies using BERT-based model"
    args_schema: type = UserAnalysisInput
    
    def __init__(self, bert_detector, data_processor):
        super().__init__()
        self._bert_detector = bert_detector
        self._data_processor = data_processor
    
    def _run(self, user_id: str) -> str:
        """Detect anomalies using BERT model."""
        try:
            if not self._bert_detector.is_fitted:
                return "BERT model not fitted. Please train the model first."
            
            # Extract user ID if present in query
            import re
            user_match = re.search(r'USER\d+', user_id)
            if user_match:
                user_id = user_match.group()
            
            # Get user's BERT sequence
            bert_sequences = self._data_processor.prepare_bert_sequences()
            
            if user_id not in bert_sequences:
                return f"No BERT sequence found for user {user_id}"
            
            # Extract BERT features
            user_embeddings = self._bert_detector.extract_bert_features({user_id: bert_sequences[user_id]})
            
            if user_id not in user_embeddings:
                return f"Failed to extract BERT features for user {user_id}"
            
            # Detect anomalies
            results = self._bert_detector.detect_anomalies(user_embeddings)
            
            if user_id not in results:
                return f"No BERT anomaly detection results for user {user_id}"
            
            result = results[user_id]
            
            response = f"BERT Anomaly Detection Results for User {user_id}:\n\n"
            response += f"Anomaly Detected: {'Yes' if result['is_anomaly'] else 'No'}\n"
            response += f"Anomaly Score: {result['anomaly_score']:.4f}\n"
            response += f"Reconstruction Error: {result['reconstruction_error']:.4f}\n"
            response += f"Embedding Norm: {result['embedding_norm']:.3f}\n\n"
            response += f"Explanation: {result['explanation']}\n\n"
            
            return response
        except Exception as e:
            return f"Error in BERT anomaly detection: {str(e)}"

class AgenticRAGAgentLocal:
    """
    Agentic RAG system for time-series analysis with local LLM support.
    Uses Gemma-2B instead of OpenAI for local inference.
    """
    
    def __init__(self, data_processor, vector_store, 
                 markov_detector, bert_detector,
                 model_type: str = "gemma-2b",
                 llm_kwargs: Optional[Dict] = None):
        self.data_processor = data_processor
        self.vector_store = vector_store
        self.markov_detector = markov_detector
        self.bert_detector = bert_detector
        
        # Initialize local LLM
        llm_kwargs = llm_kwargs or {}
        self.llm = create_local_llm(
            model_type=model_type,
            temperature=0.1,
            max_length=2048,
            use_4bit=True,
            **llm_kwargs
        )
        
        # Create tools
        self.tools = [
            TimeSeriesRetrievalTool(vector_store),
            TrendAnalysisTool(data_processor),
            MarkovAnomalyDetectionTool(markov_detector, vector_store),
            BertAnomalyDetectionTool(bert_detector, data_processor)
        ]
        
        # Create agent using initialize_agent
        self.agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent="zero-shot-react-description",
            verbose=True,
            handle_parsing_errors=True
        )
    
    def analyze_user(self, user_id: str, analysis_type: str = "both") -> str:
        """
        Analyze a specific user for potential insider threats.
        
        Args:
            user_id: User ID to analyze
            analysis_type: Type of analysis ("markov", "bert", or "both")
            
        Returns:
            Analysis results as string
        """
        query = f"Analyze potential insider threats for user {user_id} using {analysis_type} anomaly detection on logon and device data."
        
        try:
            result = self.agent_executor.invoke({"input": query})
            return result["output"]
        except Exception as e:
            return f"Error analyzing user {user_id}: {str(e)}"
    
    def search_similar_behavior(self, query: str) -> str:
        """
        Search for similar behavioral patterns.
        
        Args:
            query: Search query
            
        Returns:
            Search results as string
        """
        try:
            result = self.agent_executor.invoke({"input": query})
            return result["output"]
        except Exception as e:
            return f"Error searching for similar behavior: {str(e)}"
    
    def comprehensive_analysis(self, user_id: str) -> str:
        """
        Perform comprehensive analysis of a user.
        
        Args:
            user_id: User ID to analyze
            
        Returns:
            Comprehensive analysis results
        """
        query = f"Perform comprehensive analysis of user {user_id} including trend analysis, anomaly detection using both Markov Chain and BERT methods, and identify any potential insider threats."
        
        try:
            result = self.agent_executor.invoke({"input": query})
            return result["output"]
        except Exception as e:
            return f"Error in comprehensive analysis: {str(e)}"
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all system components."""
        status = {
            "data_processor": {
                "has_merged_data": self.data_processor.merged_data is not None,
                "has_user_features": self.data_processor.user_features is not None,
                "has_user_sequences": len(self.data_processor.user_sequences) > 0
            },
            "vector_store": {
                "has_index": self.vector_store.vector_store.index is not None,
                "index_size": self.vector_store.vector_store.index.ntotal if self.vector_store.vector_store.index else 0
            },
            "markov_detector": {
                "is_fitted": self.markov_detector.is_fitted,
                "n_clusters": self.markov_detector.n_clusters
            },
            "bert_detector": {
                "is_fitted": self.bert_detector.is_fitted,
                "model_name": self.bert_detector.model_name,
                "device": self.bert_detector.device
            },
            "agent": {
                "tools_count": len(self.tools),
                "llm_model": f"Local {self.llm.model_name}",
                "llm_device": self.llm.device
            }
        }
        
        return status

def test_local_agent():
    """Test the local agent functionality."""
    print("🧪 Testing Local Agent...")
    
    try:
        # Import required components
        from data_processor import CERTDataProcessor
        from vector_store import TimeSeriesVectorStore
        from markov_anomaly_detector import MarkovAnomalyDetector
        from bert_anomaly_detector import BertAnomalyDetector
        
        # Initialize components
        data_processor = CERTDataProcessor()
        vector_store = TimeSeriesVectorStore()
        markov_detector = MarkovAnomalyDetector()
        bert_detector = BertAnomalyDetector()
        
        # Create local agent
        agent = AgenticRAGAgentLocal(
            data_processor=data_processor,
            vector_store=vector_store,
            markov_detector=markov_detector,
            bert_detector=bert_detector,
            model_type="gemma-2b"
        )
        
        print("✅ Local agent created successfully!")
        
        # Test system status
        status = agent.get_system_status()
        print(f"📊 System status: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Local agent test failed: {e}")
        return False

if __name__ == "__main__":
    test_local_agent()
