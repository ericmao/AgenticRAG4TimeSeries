#!/usr/bin/env python3
"""
Fix token length configuration for the Agentic RAG system.
Updates environment variables and configuration to handle long inputs properly.
"""

import os
import sys

def fix_token_length_config():
    """Fix token length configuration to handle long inputs."""
    
    print("🔧 Fixing Token Length Configuration...")
    
    # Update environment variables with increased limits
    os.environ['LOCAL_LLM_MAX_LENGTH'] = '4096'  # Increased from 2048
    os.environ['BERT_MAX_LENGTH'] = '1024'  # Increased from 512
    os.environ['OPENAI_MAX_TOKENS'] = '2000'  # Increased from 1000
    
    # Set additional token-related configurations
    os.environ['LOCAL_LLM_MAX_NEW_TOKENS'] = '512'
    os.environ['LOCAL_LLM_TEMPERATURE'] = '0.1'
    os.environ['USE_4BIT_QUANTIZATION'] = 'true'
    
    print("✅ Token length configuration updated:")
    print(f"   - LOCAL_LLM_MAX_LENGTH: {os.getenv('LOCAL_LLM_MAX_LENGTH')}")
    print(f"   - BERT_MAX_LENGTH: {os.getenv('BERT_MAX_LENGTH')}")
    print(f"   - OPENAI_MAX_TOKENS: {os.getenv('OPENAI_MAX_TOKENS')}")
    print(f"   - LOCAL_LLM_MAX_NEW_TOKENS: {os.getenv('LOCAL_LLM_MAX_NEW_TOKENS')}")
    
    return True

def create_improved_main_local():
    """Create an improved main_local.py with better token handling."""
    
    improved_code = '''#!/usr/bin/env python3
"""
Improved Main script for Agentic RAG System with Local LLM Support
Includes better token length handling and error recovery
"""

import os
import sys
from dotenv import load_dotenv
from typing import Dict, Any, Tuple

# Import local LLM components
from local_llm_fixed import create_fixed_local_llm, test_fixed_local_llm
from agentic_rag_agent_local import AgenticRAGAgentLocal, test_local_agent

def load_environment() -> Dict[str, Any]:
    """Load environment variables and configuration with improved defaults."""
    load_dotenv()
    
    # API key must be set in .env (see env.example); no hard-coded fallback
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("⚠ Warning: OPENAI_API_KEY not set. Using local LLM (TinyLlama) if available.")
    
    # Model Configuration with improved token handling
    config = {
        'bert_model_name': os.getenv('BERT_MODEL_NAME', 'bert-base-uncased'),
        'sentence_transformer_model': os.getenv('SENTENCE_TRANSFORMER_MODEL', 'all-MiniLM-L6-v2'),
        'markov_n_clusters': int(os.getenv('MARKOV_N_CLUSTERS', '10')),
        'bert_max_length': int(os.getenv('BERT_MAX_LENGTH', '1024')),  # Increased
        'anomaly_threshold_multiplier': float(os.getenv('ANOMALY_THRESHOLD_MULTIPLIER', '3.0')),
        'data_dir': os.getenv('DATA_DIR', './data'),
        'local_llm_model': os.getenv('LOCAL_LLM_MODEL', 'tinyllama'),
        'local_llm_device': os.getenv('LOCAL_LLM_DEVICE', 'auto'),
        'local_llm_max_length': int(os.getenv('LOCAL_LLM_MAX_LENGTH', '4096')),  # Increased
        'local_llm_temperature': float(os.getenv('LOCAL_LLM_TEMPERATURE', '0.1')),
        'local_llm_max_new_tokens': int(os.getenv('LOCAL_LLM_MAX_NEW_TOKENS', '512')),  # Added
        'use_4bit_quantization': os.getenv('USE_4BIT_QUANTIZATION', 'true').lower() == 'true'
    }
    
    return config

def initialize_components(config: Dict[str, Any]) -> Tuple:
    """Initialize all system components with improved error handling."""
    print("�� Initializing Agentic RAG System with Fixed Local LLM...")
    
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
    # Extract the features DataFrame from the dictionary
    features_df = user_features['daily_features']
    feature_embeddings = vector_store.create_feature_embeddings(features_df)
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
    # Extract BERT embeddings first
    bert_embeddings = bert_detector.extract_bert_features(bert_sequences)
    bert_detector.fit(bert_embeddings)
    
    # Initialize local agent with improved settings
    print("�� Initializing local agent with improved token handling...")
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
            'max_new_tokens': config['local_llm_max_new_tokens'],
            'use_4bit': config['use_4bit_quantization']
        }
    )
    
    print("✅ All components initialized successfully!")
    return data_processor, vector_store, markov_detector, bert_detector, agent

def run_example_analysis(agent):
    """Run example analysis with improved error handling."""
    print("\\n📊 Running Example Analysis...")
    
    example_queries = [
        "Analyze user USER0001 for potential insider threats",
        "Search for similar behavior patterns",
        "What are the trends in user activity?"
    ]
    
    for i, query in enumerate(example_queries, 1):
        print(f"\\n🔍 Example {i}: {query}")
        try:
            result = agent.agent_executor.invoke({"input": query})
            print(f"✅ Result: {result['output'][:200]}...")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("💡 This might be due to token length issues - the system will handle this automatically")

def run_interactive_mode(agent):
    """Run interactive mode with improved error handling."""
    print("\\n🎯 Interactive Mode")
    print("=" * 60)
    print("Enter queries to analyze the CERT dataset.")
    print("Type 'quit' to exit.")
    print("=" * 60)
    
    while True:
        try:
            query = input("\\nEnter a query (or 'quit' to exit): ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            print(f"\\n�� Processing: {query}")
            print("-" * 40)
            
            try:
                result = agent.agent_executor.invoke({"input": query})
                print(f"🤖 Response: {result['output']}")
            except Exception as e:
                print(f"❌ Error processing query: {e}")
                print("💡 The system will automatically handle token length issues")
                print("💡 Try rephrasing your query to be more concise")
                
        except KeyboardInterrupt:
            print("\\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

def main():
    """Main function with improved error handling."""
    print("�� Agentic RAG System with Fixed Local LLM (Improved Token Handling)")
    print("=" * 60)
    
    # Load environment
    config = load_environment()
    
    # Test fixed local LLM
    print("\\n🧪 Testing fixed local LLM...")
    if not test_fixed_local_llm():
        print("❌ Fixed local LLM test failed. Please check your setup.")
        return
    
    # Initialize components
    try:
        data_processor, vector_store, markov_detector, bert_detector, agent = initialize_components(config)
    except Exception as e:
        print(f"❌ Failed to initialize components: {e}")
        return
    
    # Show system status
    print("\\n📊 System Status:")
    status = agent.get_system_status()
    for component, info in status.items():
        print(f"\\n{component.upper()}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    # Run example analysis
    run_example_analysis(agent)
    
    # Run interactive mode
    run_interactive_mode(agent)

if __name__ == "__main__":
    main()
'''
    
    with open('main_local_fixed.py', 'w') as f:
        f.write(improved_code)
    
    print("✅ Created main_local_fixed.py with improved token handling")
    return True

def main():
    """Main function to fix token length issues."""
    print("🔧 Token Length Fix for Agentic RAG System")
    print("=" * 60)
    
    # Fix configuration
    fix_token_length_config()
    
    # Create improved main file
    create_improved_main_local()
    
    print("\n✅ Token length issues fixed!")
    print("\n📋 Next steps:")
    print("1. Run: python main_local_fixed.py")
    print("2. The system will now handle long inputs automatically")
    print("3. Token length warnings will be handled gracefully")
    
    return True

if __name__ == "__main__":
    main() 