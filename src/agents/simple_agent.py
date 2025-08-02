#!/usr/bin/env python3
"""
Simplified Agentic RAG Agent
Avoids complex Pydantic tool structures that might cause issues.
"""

import os
import sys
from typing import Dict, List, Any, Optional
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain.schema import BaseMessage, HumanMessage
import json
from datetime import datetime

# Set up environment
os.environ['OPENAI_API_KEY'] = "your_openai_api_key_here"

class SimpleAnalysisTool(BaseTool):
    """Simple analysis tool that avoids complex Pydantic structures."""
    
    def __init__(self, name: str, description: str, analysis_function):
        super().__init__()
        self.name = name
        self.description = description
        self.analysis_function = analysis_function
    
    def _run(self, query: str) -> str:
        """Run the analysis function."""
        try:
            return self.analysis_function(query)
        except Exception as e:
            return f"Error in {self.name}: {str(e)}"

class SimpleAgenticRAGAgent:
    """
    Simplified Agentic RAG system that avoids complex tool structures.
    """
    
    def __init__(self, data_processor=None, vector_store=None, 
                 markov_detector=None, bert_detector=None):
        self.data_processor = data_processor
        self.vector_store = vector_store
        self.markov_detector = markov_detector
        self.bert_detector = bert_detector
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Create simple tools
        self.tools = [
            SimpleAnalysisTool(
                name="time_series_analysis",
                description="Analyze time-series data and detect anomalies",
                analysis_function=self._analyze_time_series
            ),
            SimpleAnalysisTool(
                name="user_analysis", 
                description="Analyze specific user behavior patterns",
                analysis_function=self._analyze_user
            ),
            SimpleAnalysisTool(
                name="anomaly_detection",
                description="Detect anomalies using multiple methods",
                analysis_function=self._detect_anomalies
            )
        ]
        
        # Create agent
        self.agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            verbose=True
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def _analyze_time_series(self, query: str) -> str:
        """Analyze time-series data."""
        return f"Time-series analysis for query: {query}\n" + \
               "This would analyze trends, patterns, and anomalies in the time-series data."
    
    def _analyze_user(self, query: str) -> str:
        """Analyze user behavior."""
        return f"User behavior analysis for query: {query}\n" + \
               "This would analyze specific user patterns and activities."
    
    def _detect_anomalies(self, query: str) -> str:
        """Detect anomalies."""
        return f"Anomaly detection for query: {query}\n" + \
               "This would use both Markov Chain and BERT methods to detect anomalies."
    
    def analyze_query(self, query: str) -> str:
        """Analyze a query using the agent."""
        try:
            result = self.agent_executor.invoke({"input": query})
            return result["output"]
        except Exception as e:
            return f"Error analyzing query: {str(e)}"
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of system components."""
        status = {
            "agent": {
                "tools_count": len(self.tools),
                "llm_model": "gpt-3.5-turbo",
                "status": "ready"
            }
        }
        
        if self.data_processor:
            status["data_processor"] = {
                "has_merged_data": self.data_processor.merged_data is not None,
                "has_user_features": self.data_processor.user_features is not None
            }
        
        if self.vector_store:
            status["vector_store"] = {
                "has_index": self.vector_store.vector_store.index is not None if hasattr(self.vector_store, 'vector_store') else False
            }
        
        return status

def run_simple_agent():
    """Run the simplified agentic system."""
    print("🎯 Simple Agentic RAG System")
    print("=" * 50)
    print("This version avoids complex Pydantic structures.")
    print("=" * 50)
    
    try:
        # Initialize agent
        agent = SimpleAgenticRAGAgent()
        
        print("✅ Agent initialized successfully!")
        
        # Test the agent
        print("\n🧪 Testing agent with sample queries...")
        
        test_queries = [
            "Analyze potential insider threats for user USER0001",
            "What are the trends in user activity?",
            "Detect anomalies in the dataset"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            result = agent.analyze_query(query)
            print(f"Result: {result[:200]}...")  # Truncate for display
        
        # Interactive mode
        print("\n" + "=" * 40)
        print("INTERACTIVE MODE")
        print("=" * 40)
        print("Enter 'quit' to exit")
        
        while True:
            try:
                user_input = input("\nEnter a query (or 'quit' to exit): ").strip()
                
                if user_input.lower() == 'quit':
                    break
                
                if not user_input:
                    continue
                
                # Process the query
                result = agent.analyze_query(user_input)
                print("\n" + "=" * 40)
                print("RESULT")
                print("=" * 40)
                print(result)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error processing query: {str(e)}")
        
        print("\n👋 Simple agent completed!")
        
    except Exception as e:
        print(f"❌ Error initializing agent: {str(e)}")
        print("This might be due to missing dependencies or API issues.")

if __name__ == "__main__":
    run_simple_agent() 