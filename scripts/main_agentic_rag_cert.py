#!/usr/bin/env python3
"""
Core Agentic RAG System for CERT Anomaly Analysis
Focused on essential components: Data Processing, Vector Store, Anomaly Detection, and LLM Analysis
"""

import os
import sys
import time
import numpy as np
from datetime import datetime

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_openai():
    """Setup OpenAI configuration."""
    print("🔧 Setting up OpenAI...")
    
    # Set the API key from environment variable or prompt user
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("⚠️ OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return False
    
    os.environ['OPENAI_API_KEY'] = api_key
    print("✅ OpenAI API key configured")
    return True

def test_openai_connection():
    """Test OpenAI connection."""
    print("\n🧪 Testing OpenAI Connection...")
    
    try:
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            max_tokens=500
        )
        
        test_prompt = "Explain anomaly detection in cybersecurity in one sentence."
        
        print(f"📝 Testing with prompt: {test_prompt}")
        start_time = time.time()
        
        response = llm.invoke(test_prompt)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"✅ Response received in {response_time:.2f}s")
        print(f"📄 Response: {response.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")
        return False

def initialize_agentic_rag_system():
    """Initialize the core Agentic RAG system."""
    print("\n🚀 Initializing Core Agentic RAG System...")
    
    try:
        # Import core components
        from src.core.data_processor import CERTDataProcessor
        from src.core.vector_store import VectorStore
        from src.core.markov_anomaly_detector import MarkovAnomalyDetector
        from src.core.bert_anomaly_detector import BertAnomalyDetector
        from langchain_openai import ChatOpenAI
        from src.utils.model_persistence import (
            should_use_existing_models, 
            load_existing_models, 
            print_model_status,
            save_trained_models
        )
        
        # Initialize data processor
        print("📊 Initializing CERT data processor...")
        data_processor = CERTDataProcessor()
        
        # Load and process CERT data
        print("📈 Loading and processing CERT data...")
        data_processor.load_sample_data()
        data_processor.engineer_features()
        data_processor.textualize_events()
        data_processor.prepare_bert_sequences()
        
        # Initialize vector store
        print("🔍 Initializing vector store...")
        vector_store = VectorStore()
        
        # Create embeddings
        if hasattr(data_processor, 'user_sequences') and data_processor.user_sequences:
            texts = list(data_processor.user_sequences.values())
            metadata = [{'user': user_id, 'sequence_length': len(seq)} for user_id, seq in data_processor.user_sequences.items()]
            vector_store.create_embeddings(texts, metadata)
            vector_store.build_index(np.array(vector_store.embeddings))
        else:
            print("⚠️ No user sequences available for vector store")
        
        # Check for existing models
        print("📂 Checking for existing trained models...")
        use_existing, model_status = should_use_existing_models(
            force_retrain=False,
            max_model_age_days=30
        )
        
        print_model_status(model_status)
        
        # Initialize model detectors
        markov_detector = None
        bert_detector = None
        
        if use_existing:
            print("📥 Loading existing models...")
            markov_detector, bert_detector = load_existing_models(model_status)
            if markov_detector and bert_detector:
                print("✅ Successfully loaded existing models.")
            else:
                print("❌ Failed to load all existing models. Will train new ones.")
                use_existing = False
        
        if not use_existing:
            print("🔄 Training new models...")
            
            # Train Markov Chain detector
            print("🔗 Training Markov Chain detector...")
            markov_detector = MarkovAnomalyDetector()
            
            if hasattr(data_processor, 'user_features') and data_processor.user_features is not None:
                # Filter numeric columns and clean data
                numeric_columns = []
                for col in data_processor.user_features.columns:
                    if col != 'user':
                        if data_processor.user_features[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                            numeric_columns.append(col)
                
                if numeric_columns:
                    clean_features = data_processor.user_features[numeric_columns + ['user']].dropna()
                    
                    if len(clean_features) > 0:
                        user_embeddings_dict = {}
                        
                        for _, row in clean_features.iterrows():
                            user_id = row['user']
                            features = row[numeric_columns].values.reshape(1, -1)
                            user_embeddings_dict[user_id] = features
                        
                        if user_embeddings_dict:
                            markov_detector.fit(user_embeddings_dict)
                            print(f"✅ Markov Chain model trained successfully with {len(numeric_columns)} features")
                        else:
                            print("⚠️ No valid user embeddings for Markov training")
                    else:
                        print("⚠️ No clean data available for Markov training")
                else:
                    print("⚠️ No numeric features available for Markov training")
            else:
                print("⚠️ No user features available for Markov training")
            
            # Train BERT detector
            print("🤖 Training BERT detector...")
            bert_detector = BertAnomalyDetector()
            
            if hasattr(data_processor, 'user_sequences') and data_processor.user_sequences:
                try:
                    # Prepare BERT sequences in the correct format
                    bert_sequences = data_processor.prepare_bert_sequences()
                    
                    if bert_sequences:
                        # Extract BERT features
                        bert_features = bert_detector.extract_bert_features(bert_sequences)
                        
                        if bert_features:
                            # Fit the detector with extracted features
                            bert_detector.fit(bert_features)
                            print("✅ BERT model trained successfully")
                        else:
                            print("⚠️ No valid BERT features extracted")
                    else:
                        print("⚠️ No valid BERT sequences available")
                except Exception as e:
                    print(f"⚠️ BERT training failed: {e}")
                    bert_detector = None
            else:
                print("⚠️ No user sequences available for BERT training")
            
            # Save trained models
            if markov_detector or bert_detector:
                print("💾 Saving trained models...")
                save_trained_models(markov_detector, bert_detector)
                print("✅ Models saved successfully")
        
        # Create LLM
        print("🤖 Creating GPT-4o LLM...")
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            max_tokens=1000
        )
        
        print("✅ Core Agentic RAG system initialized successfully!")
        
        return {
            "data_processor": data_processor,
            "vector_store": vector_store,
            "markov_detector": markov_detector,
            "bert_detector": bert_detector,
            "llm": llm
        }
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return None

def print_system_status(components):
    """Print system status."""
    print("\n📊 Core Agentic RAG System Status:")
    print("=" * 60)
    
    dp = components["data_processor"]
    vs = components["vector_store"]
    markov_detector = components.get("markov_detector")
    bert_detector = components.get("bert_detector")
    llm = components["llm"]
    
    print("DATA_PROCESSOR:")
    print(f"  has_merged_data: {dp.merged_data is not None}")
    print(f"  has_user_features: {dp.user_features is not None}")
    print(f"  has_user_sequences: {dp.user_sequences is not None}")
    
    print("\nVECTOR_STORE:")
    print(f"  has_index: {vs.index is not None}")
    print(f"  index_size: {vs.index.ntotal if vs.index is not None else 'N/A'}")
    
    print("\nANOMALY_DETECTORS:")
    print(f"  markov_detector: {'✅ Available' if markov_detector else '❌ Not available'}")
    print(f"  bert_detector: {'✅ Available' if bert_detector else '❌ Not available'}")
    
    print("\nLLM:")
    print(f"  model: GPT-4o")
    print(f"  type: Cloud-based")

def main():
    """Main function."""
    print("🔍 Core Agentic RAG for CERT Anomaly Analysis")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup OpenAI
    if not setup_openai():
        print("\n❌ OpenAI setup failed.")
        return
    
    # Test OpenAI connection
    if not test_openai_connection():
        print("\n❌ OpenAI connection test failed.")
        return
    
    # Initialize system
    components = initialize_agentic_rag_system()
    if not components:
        print("\n❌ System initialization failed.")
        return
    
    # Print system status
    print_system_status(components)
    
    print(f"\n🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 