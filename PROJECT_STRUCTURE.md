# Agentic RAG Project Structure

## Core Components

### Scripts
- `scripts/main_agentic_rag_cert.py` - Core Agentic RAG system for CERT analysis
- `scripts/main_gpt4o_cert_simple.py` - GPT-4o based CERT analysis
- `scripts/main.py` - Original main script

### Source Code
- `src/core/` - Core components (data processor, vector store, anomaly detectors)
- `src/utils/` - Utility functions (model persistence)
- `src/agents/` - Agent components

### Data and Models
- `data/` - Data files (CERT dataset)
- `models/` - Trained models (Markov, BERT)
- `docs/` - Documentation

### Configuration
- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore rules
- `README.md` - Project documentation

## Key Features

1. **Multi-modal Anomaly Detection**
   - Markov Chain analysis
   - BERT-based text analysis
   - Vector similarity search

2. **GPT-4o Integration**
   - Intelligent analysis
   - Risk assessment
   - Detailed reporting

3. **Model Persistence**
   - Automatic model saving/loading
   - Version management
   - Training optimization

4. **CERT Dataset Analysis**
   - Real insider threat data
   - Behavioral pattern analysis
   - Security risk assessment
