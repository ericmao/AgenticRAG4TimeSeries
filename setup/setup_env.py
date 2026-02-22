#!/usr/bin/env python3
"""
Setup script to configure the environment with API keys and settings.
"""

import os
import sys

def setup_environment():
    """Load env from .env (copy from env.example) and set defaults for optional vars."""
    from pathlib import Path
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k and v and k not in os.environ:
                        os.environ[k] = v
    # Optional defaults (do not set secrets)
    os.environ.setdefault('BERT_MODEL_NAME', 'bert-base-uncased')
    os.environ.setdefault('SENTENCE_TRANSFORMER_MODEL', 'all-MiniLM-L6-v2')
    os.environ.setdefault('MARKOV_N_CLUSTERS', '10')
    os.environ.setdefault('BERT_MAX_LENGTH', '512')
    os.environ.setdefault('ANOMALY_THRESHOLD_MULTIPLIER', '3.0')
    os.environ.setdefault('DATA_DIR', './data')
    print("✅ Environment loaded from .env (if present) and defaults set.")
    if os.environ.get('OPENAI_API_KEY'):
        v = os.environ['OPENAI_API_KEY']
        print(f"✅ OPENAI_API_KEY: {v[:8]}...{v[-4:] if len(v) > 12 else '(set)'}")
    else:
        print("⚠ OPENAI_API_KEY not set. Copy env.example to .env and fill in.")
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