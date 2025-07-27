# Agentic RAG System for Time-Series Analysis

## Overview

This project implements an advanced Agentic RAG (Retrieval-Augmented Generation) system specifically designed for time-series analysis on the CERT Insider Threat dataset. The system combines multiple anomaly detection approaches with LangChain-based agentic workflows to identify potential insider threats.

## Key Features

### 🔍 **Dual Anomaly Detection**
- **Markov Chain Model**: Uses K-means clustering to discretize embeddings into states, builds transition probability matrices, and detects anomalies through sequence likelihood analysis
- **BERT-based Model**: Leverages pre-trained BERT transformers with autoencoder architecture for sequence-level anomaly detection

### 🧠 **Agentic Workflows**
- LangChain-powered agent with 4 specialized tools:
  - Time-series sequence retrieval
  - Trend analysis with rolling statistics
  - Markov Chain anomaly detection
  - BERT-based anomaly detection

### 📊 **Time-Series Processing**
- Comprehensive feature engineering (daily windows, rolling statistics, anomaly scores)
- Event textualization for embedding generation
- Vector similarity search using FAISS

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Loading  │    │  Feature        │    │  Vector Store   │
│   & Processing  │───▶│  Engineering    │───▶│  & Embeddings   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  LangChain      │    │  Markov Chain   │    │  BERT-based     │
│  Agent          │◀───│  Anomaly        │    │  Anomaly        │
│  (4 Tools)      │    │  Detector       │    │  Detector       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key
- CUDA-compatible GPU (optional, for BERT acceleration)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd AgenticRAG
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp env_example.txt .env
```

Edit `.env` file with your configuration:
```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Model Configuration
BERT_MODEL_NAME=bert-base-uncased
SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2

# Anomaly Detection Parameters
MARKOV_N_CLUSTERS=10
BERT_MAX_LENGTH=512
ANOMALY_THRESHOLD_MULTIPLIER=3.0

# Data Configuration
DATA_DIR=./data
```

4. **Prepare data directory** (optional)
```bash
mkdir data
# Place CERT dataset CSV files (logon.csv, device.csv) in data/ directory
```

## Usage

### Quick Start

Run the main script to initialize and test the system:

```bash
python main.py
```

The system will:
1. Load and process the CERT dataset (creates synthetic data if actual files not found)
2. Engineer time-series features
3. Build vector embeddings and index
4. Train both anomaly detection models
5. Run example analysis
6. Enter interactive mode for queries

### Interactive Mode

Once initialized, you can interact with the system using natural language queries:

```
Enter a query (or 'quit' to exit): Analyze potential insider threats for user USER0001 using both Markov chain and BERT-based anomaly detection
```

### Example Queries

- `"Analyze user USER0001 for anomalies"`
- `"Search for similar behavior patterns to unusual logon times"`
- `"What are the trends in user USER0002's activity over the last week?"`
- `"Compare Markov and BERT results for user USER0003"`

## System Components

### 1. Data Processor (`data_processor.py`)
- Loads and merges CERT dataset files
- Performs time-series feature engineering
- Textualizes events for embedding generation
- Prepares BERT sequences

### 2. Vector Store (`vector_store.py`)
- Manages FAISS-based similarity search
- Creates embeddings using Sentence Transformers
- Handles both sequence and feature embeddings

### 3. Markov Chain Detector (`markov_anomaly_detector.py`)
- Discretizes embeddings using K-means clustering
- Builds transition probability matrices
- Detects anomalies through sequence likelihood analysis
- Uses Isolation Forest for outlier detection

### 4. BERT Anomaly Detector (`bert_anomaly_detector.py`)
- Extracts features using pre-trained BERT
- Implements autoencoder for reconstruction-based anomaly detection
- Handles variable-length sequences
- Provides detailed embedding analysis

### 5. Agentic RAG Agent (`agentic_rag_agent.py`)
- LangChain agent with 4 specialized tools
- Handles natural language queries
- Orchestrates multiple analysis methods
- Provides comprehensive insights

## Configuration

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MARKOV_N_CLUSTERS` | 10 | Number of states for Markov Chain |
| `BERT_MAX_LENGTH` | 512 | Maximum sequence length for BERT |
| `ANOMALY_THRESHOLD_MULTIPLIER` | 3.0 | Threshold multiplier for anomaly detection |

### Training Parameters

- **Markov Chain**: Uses 80% of users for training, 20% for testing
- **BERT Autoencoder**: 10 epochs, learning rate 1e-3, batch size 32
- **Isolation Forest**: Contamination rate 0.1 (10% assumed anomalous)

## Output Examples

### Anomaly Detection Results

```
Markov Chain Anomaly Detection Results for User USER0001:

Anomaly Detected: Yes
Anomaly Score: -0.8234
Sequence Likelihood: -12.4567
Sequence Length: 45
Unique States: 8

Explanation: Anomaly detected in sequence with low likelihood (-12.4567). 
Sequence length: 45, Unique transitions: 12. Very unusual transition patterns detected.

BERT Anomaly Detection Results for User USER0001:

Anomaly Detected: Yes
Anomaly Score: -0.9123
Reconstruction Error: 0.1567
Embedding Norm: 2.345

Explanation: Anomaly detected with high reconstruction error (0.1567). 
Embedding norm: 2.345, Embedding sparsity: 0.234. 
Very high reconstruction error indicates unusual sequence patterns.
```

### Trend Analysis

```
Trend Analysis for User USER0001 (Window: 7 days):

Total Activities:
  Current Value: 15.00
  Rolling Mean: 12.34
  Trend: 0.234 (increasing)

Logon Count:
  Current Value: 8.00
  Rolling Mean: 6.78
  Trend: 0.123 (increasing)
```

## Performance Considerations

### Memory Usage
- BERT model: ~500MB
- FAISS index: ~100MB per 10K vectors
- Sentence Transformers: ~100MB

### Processing Time
- Data processing: 2-5 minutes for 10K records
- BERT training: 5-10 minutes (10 epochs)
- Markov training: 1-2 minutes
- Query processing: 1-5 seconds

### GPU Acceleration
- BERT model automatically uses GPU if available
- Set `CUDA_VISIBLE_DEVICES` for multi-GPU setups
- CPU fallback available for all components

## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   ```
   ERROR: OPENAI_API_KEY not found in environment variables.
   ```
   Solution: Set your OpenAI API key in the `.env` file

2. **CUDA Out of Memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   Solution: Reduce batch size or use CPU-only mode

3. **FAISS Index Error**
   ```
   ValueError: Index not built. Call build_index() first.
   ```
   Solution: Ensure data processing completed successfully

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- CERT Insider Threat Dataset from Kaggle
- LangChain for agentic workflows
- Hugging Face Transformers for BERT implementation
- FAISS for efficient similarity search
- Sentence Transformers for embedding generation

## Citation

If you use this system in your research, please cite:

```bibtex
@software{agentic_rag_insider_threat,
  title={Agentic RAG System for Time-Series Analysis on CERT Insider Threat Dataset},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/AgenticRAG}
}
``` 