import os
from typing import Dict, List, Any, Optional
from langchain.agents import AgentExecutor, create_openai_tools_agent, initialize_agent
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain.schema import BaseMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
import json
from datetime import datetime

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
            
            response = f"Trend Analysis for User {user_id} (Window: 7 days):\n\n"
            for metric, data in trends.items():
                response += f"{metric.replace('_', ' ').title()}:\n"
                response += f"  Current Value: {data['current_value']:.2f}\n"
                response += f"  Rolling Mean: {data['rolling_mean']:.2f}\n"
                response += f"  Trend: {data['trend']:.3f} ({data['trend_direction']})\n\n"
            
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
            
            # Add transition analysis
            if result['state_transitions']:
                transitions = result['state_transitions']
                response += f"Transition Analysis:\n"
                response += f"  Total Transitions: {transitions.get('total_transitions', 0)}\n"
                response += f"  Unique Transitions: {transitions.get('unique_transitions', 0)}\n"
                response += f"  Transition Entropy: {transitions.get('transition_entropy', 0):.3f}\n"
                
                if transitions.get('most_common_transition'):
                    most_common = transitions['most_common_transition']
                    response += f"  Most Common Transition: {most_common[0]} (count: {most_common[1]})\n"
            
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
            
            # Add embedding analysis
            if result['embedding_analysis']:
                analysis = result['embedding_analysis']
                response += f"Embedding Analysis:\n"
                response += f"  Mean: {analysis['mean']:.4f}\n"
                response += f"  Standard Deviation: {analysis['std']:.4f}\n"
                response += f"  Max Value: {analysis['max_value']:.4f}\n"
                response += f"  Min Value: {analysis['min_value']:.4f}\n"
                response += f"  Sparsity: {analysis['sparsity']:.3f}\n"
            
            return response
        except Exception as e:
            return f"Error in BERT anomaly detection: {str(e)}"

class AgenticRAGAgent:
    """
    Agentic RAG system for time-series analysis and insider threat detection.
    """
    
    def __init__(self, openai_api_key: str, data_processor, vector_store, 
                 markov_detector, bert_detector):
        self.data_processor = data_processor
        self.vector_store = vector_store
        self.markov_detector = markov_detector
        self.bert_detector = bert_detector
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        
        # Create tools
        self.tools = [
            TimeSeriesRetrievalTool(vector_store),
            TrendAnalysisTool(data_processor),
            MarkovAnomalyDetectionTool(markov_detector, vector_store),
            BertAnomalyDetectionTool(bert_detector, data_processor)
        ]
        
        # Create agent using initialize_agent (more compatible)
        self.agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent="zero-shot-react-description",
            verbose=True,
            handle_parsing_errors=True
        )
        
        # Agent executor is already created above
    
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
            query: Query describing the behavior pattern
            
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
        Perform comprehensive analysis including both anomaly detection methods.
        
        Args:
            user_id: User ID to analyze
            
        Returns:
            Comprehensive analysis results
        """
        query = f"""
        Perform a comprehensive analysis of user {user_id} for insider threat detection:
        1. Retrieve similar time-series sequences for this user
        2. Analyze trends in user behavior
        3. Run Markov Chain anomaly detection
        4. Run BERT-based anomaly detection
        5. Compare results and provide insights
        """
        
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
                "llm_model": "gpt-3.5-turbo"
            }
        }
        
        return status 