#!/usr/bin/env python3
"""
Setup script to configure the environment with API keys and settings.
"""

import os
import sys

def setup_environment():
    """Set up the environment variables for the Agentic RAG system."""
    
    # API Key configuration (replace with your own key)
    api_key = "your_openai_api_key_here"
    
    # Set environment variables
    os.environ['OPENAI_API_KEY'] = api_key
    os.environ['BERT_MODEL_NAME'] = 'bert-base-uncased'
    os.environ['SENTENCE_TRANSFORMER_MODEL'] = 'all-MiniLM-L6-v2'
    os.environ['MARKOV_N_CLUSTERS'] = '10'
    os.environ['BERT_MAX_LENGTH'] = '512'
    os.environ['ANOMALY_THRESHOLD_MULTIPLIER'] = '3.0'
    os.environ['DATA_DIR'] = './data'
    
    print("✅ Environment configured successfully!")
    print(f"✅ OpenAI API Key: {api_key[:20]}...{api_key[-10:]}")
    print("✅ All environment variables set")
    
    return True

def test_environment():
    """Test that the environment is properly configured."""
    print("\n🔍 Testing environment configuration...")
    
    # Check required environment variables
    required_vars = [
        'OPENAI_API_KEY',
        'BERT_MODEL_NAME', 
        'SENTENCE_TRANSFORMER_MODEL',
        'MARKOV_N_CLUSTERS',
        'BERT_MAX_LENGTH',
        'ANOMALY_THRESHOLD_MULTIPLIER',
        'DATA_DIR'
    ]
    
    all_good = True
    for var in required_vars:
        value = os.environ.get(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set")
            all_good = False
    
    if all_good:
        print("\n🎉 Environment is ready!")
        return True
    else:
        print("\n❌ Environment configuration incomplete")
        return False

if __name__ == "__main__":
    print("🚀 Setting up Agentic RAG Environment")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Test environment
    if test_environment():
        print("\n✅ Setup complete! You can now run:")
        print("   python test_system.py  # Test components")
        print("   python main.py         # Run full system")
        print("   python example_usage.py # See examples")
    else:
        print("\n❌ Setup failed. Please check the configuration.")
        sys.exit(1) 