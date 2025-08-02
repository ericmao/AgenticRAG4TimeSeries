#!/usr/bin/env python3
"""
Main script for Agentic RAG System with Local LLM Support
Uses Gemma-2B instead of OpenAI for local inference
"""

import os
import sys
from dotenv import load_dotenv
from typing import Dict, Any, Tuple

# Import local LLM components
from local_llm import create_local_llm, test_local_llm
from agentic_rag_agent_local import AgenticRAGAgentLocal, test_local_agent

def load_environment() -> Dict[str, Any]:
    """Load environment variables and configuration."""
    load_dotenv()
    
    # Set up environment variables if not already set
    if not os.getenv('OPENAI_API_KEY'):
        os.environ['OPENAI_API_KEY'] = "your_openai_api_key_here"
        print("✅ OpenAI API key configured automatically")
    
    # Check for required API key
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key or openai_api_key == "your_openai_api_key_here":
        print("⚠ Warning: Using local LLM (Gemma-2B) instead of OpenAI")
    
    # Model Configuration
    config = {
        'bert_model_name': os.getenv('BERT_MODEL_NAME', 'bert-base-uncased'),
        'sentence_transformer_model': os.getenv('SENTENCE_TRANSFORMER_MODEL', 'all-MiniLM-L6-v2'),
        'markov_n_clusters': int(os.getenv('MARKOV_N_CLUSTERS', '10')),
        'bert_max_length': int(os.getenv('BERT_MAX_LENGTH', '512')),
        'anomaly_threshold_multiplier': float(os.getenv('ANOMALY_THRESHOLD_MULTIPLIER', '3.0')),
        'data_dir': os.getenv('DATA_DIR', './data'),
        'local_llm_model': os.getenv('LOCAL_LLM_MODEL', 'gemma-2b'),
        'local_llm_device': os.getenv('LOCAL_LLM_DEVICE', 'auto'),
        'local_llm_max_length': int(os.getenv('LOCAL_LLM_MAX_LENGTH', '2048')),
        'local_llm_temperature': float(os.getenv('LOCAL_LLM_TEMPERATURE', '0.1')),
        'use_4bit_quantization': os.getenv('USE_4BIT_QUANTIZATION', 'true').lower() == 'true'
    }
    
    return config

def initialize_components(config: Dict[str, Any]) -> Tuple:
    """Initialize all system components."""
    print("🚀 Initializing Agentic RAG System with Local LLM...")
    
    # Import components
    from data_processor import CERTDataProcessor
    from vector_store import TimeSeriesVectorStore
    from markov_anomaly_detector import MarkovAnomalyDetector
    from bert_anomaly_detector import BertAnomalyDetector
    
    # Initialize data processor
    print("📊 Initializing data processor...")
    data_processor = CERTDataProcessor(data_dir=config['data_dir'])
    
    # Load and process data
    print("📈 Loading and processing data...")
    merged_data = data_processor.load_sample_data()
    user_features = data_processor.engineer_features()
    user_sequences = data_processor.textualize_events()
    bert_sequences = data_processor.prepare_bert_sequences()
    
    print(f"✅ Processed {len(merged_data)} records for {merged_data['user'].nunique()} users")
    
    # Initialize vector store
    print("🔍 Initializing vector store...")
    vector_store = TimeSeriesVectorStore(model_name=config['sentence_transformer_model'])
    
    # Create embeddings
    print("📝 Creating embeddings...")
    feature_embeddings = vector_store.create_feature_embeddings(user_features)
    sequence_embeddings = vector_store.create_sequence_embeddings(user_sequences)
    
    print(f"✅ Created embeddings for {len(feature_embeddings)} users")
    
    # Initialize Markov detector
    print("🔗 Initializing Markov Chain detector...")
    markov_detector = MarkovAnomalyDetector(n_clusters=config['markov_n_clusters'])
    
    # Train Markov detector
    print("🎯 Training Markov Chain detector...")
    markov_detector.fit(feature_embeddings)
    
    # Initialize BERT detector
    print("🤖 Initializing BERT detector...")
    bert_detector = BertAnomalyDetector(
        model_name=config['bert_model_name'],
        max_length=config['bert_max_length']
    )
    
    # Train BERT detector
    print("🎯 Training BERT detector...")
    bert_detector.fit(bert_sequences)
    
    # Initialize local agent
    print("🤖 Initializing local agent with Gemma-2B...")
    agent = AgenticRAGAgentLocal(
        data_processor=data_processor,
        vector_store=vector_store,
        markov_detector=markov_detector,
        bert_detector=bert_detector,
        model_type=config['local_llm_model'],
        llm_kwargs={
            'device': config['local_llm_device'],
            'max_length': config['local_llm_max_length'],
            'temperature': config['local_llm_temperature'],
            'use_4bit': config['use_4bit_quantization']
        }
    )
    
    print("✅ All components initialized successfully!")
    return data_processor, vector_store, markov_detector, bert_detector, agent

def run_example_analysis(agent: AgenticRAGAgentLocal):
    """Run example analysis to demonstrate the system."""
    print("\n" + "=" * 60)
    print("🔍 EXAMPLE ANALYSIS")
    print("=" * 60)
    
    # Example 1: Analyze a specific user
    print("\n📋 Example 1: User Analysis")
    print("-" * 40)
    result = agent.analyze_user("USER0001", "both")
    print(result)
    
    # Example 2: Search for similar behavior
    print("\n🔍 Example 2: Similar Behavior Search")
    print("-" * 40)
    result = agent.search_similar_behavior("Find users with similar late-night login patterns")
    print(result)
    
    # Example 3: Comprehensive analysis
    print("\n📋 Example 3: Comprehensive Analysis")
    print("-" * 40)
    result = agent.comprehensive_analysis("USER0005")
    print(result)

def run_interactive_mode(agent: AgenticRAGAgentLocal):
    """Run interactive mode for user queries."""
    print("\n" + "=" * 60)
    print("🎮 INTERACTIVE MODE")
    print("=" * 60)
    print("Ask questions about user behavior, anomalies, or trends.")
    print("Type 'quit' to exit.")
    print("=" * 60)
    
    while True:
        try:
            query = input("\nEnter a query (or 'quit' to exit): ").strip()
            
            if query.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            print("\n🤖 Processing query...")
            result = agent.agent_executor.invoke({"input": query})
            print(f"\n📝 Response: {result['output']}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error processing query: {str(e)}")

def main():
    """Main function."""
    print("🤖 Agentic RAG System with Local LLM (Gemma-2B)")
    print("=" * 60)
    
    # Load environment
    config = load_environment()
    
    # Test local LLM
    print("\n🧪 Testing local LLM...")
    if not test_local_llm():
        print("❌ Local LLM test failed. Please check your setup.")
        return
    
    # Initialize components
    try:
        data_processor, vector_store, markov_detector, bert_detector, agent = initialize_components(config)
    except Exception as e:
        print(f"❌ Failed to initialize components: {e}")
        return
    
    # Show system status
    print("\n📊 System Status:")
    status = agent.get_system_status()
    for component, info in status.items():
        print(f"\n{component.upper()}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    # Run example analysis
    run_example_analysis(agent)
    
    # Run interactive mode
    run_interactive_mode(agent)

if __name__ == "__main__":
    main()
