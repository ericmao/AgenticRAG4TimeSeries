# 🚀 Quick Start Guide

## Configure environment (no secrets in repo)

Copy `env.example` to `.env` in the repo root and set `OPENAI_API_KEY` (and any other variables you need). Never commit `.env`. For dry-run validation without external services, you can leave keys empty and set `RUN_MODE=dry_run`.

## 🎯 Multiple Setup Options

### Option 1: Full System (Recommended)
Run the quick start script to automatically set up everything:

```bash
python quick_start.py
```

This will:
1. ✅ Configure your API key automatically
2. 📦 Install all dependencies
3. 🧪 Test system components
4. 🎯 Launch the full Agentic RAG system

### Option 2: Robust Installation
If you encounter dependency issues, use the robust installer:

```bash
python install_dependencies.py
```

This will install dependencies in stages with better error handling.

### Option 3: Minimal System (Fallback)
If you have dependency issues, try the minimal version:

```bash
python minimal_system.py
```

This works with basic dependencies only (pandas, numpy) and provides core functionality.

### Option 4: Simple Agent (LangChain Issues)
If you encounter Pydantic/LangChain issues, try the simple agent:

```bash
python simple_agent.py
```

This avoids complex tool structures and should work with basic LangChain setup.

## 🛠️ Manual Setup (Alternative)

If you prefer manual setup:

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test the System
```bash
python test_system.py
```

### 3. Run the Main System
```bash
python main.py
```

## 🎮 Interactive Usage

Once the system is running, you can ask questions like:

- `"Analyze potential insider threats for user USER0001"`
- `"Search for similar behavior patterns to unusual logon times"`
- `"What are the trends in user USER0002's activity?"`
- `"Compare Markov and BERT results for user USER0003"`

## 📊 What You'll See

The system will:
1. **Load Data**: Process CERT Insider Threat dataset (creates synthetic data if needed)
2. **Engineer Features**: Create time-series features and anomaly scores
3. **Build Embeddings**: Generate vector embeddings for similarity search
4. **Train Models**: Fit both Markov Chain and BERT anomaly detectors
5. **Interactive Mode**: Allow natural language queries

## 🔍 Example Output

```
Markov Chain: Anomaly detected (Score: -0.8234, Likelihood: -12.4567)
BERT: Anomaly detected (Score: -0.9123, Reconstruction Error: 0.1567)
Explanation: Very unusual transition patterns detected with high reconstruction error
```

## 🎯 System Features

- ✅ **Dual Anomaly Detection**: Markov Chain + BERT approaches
- ✅ **Time-Series Analysis**: Rolling statistics and trend detection
- ✅ **Vector Similarity Search**: Find similar behavioral patterns
- ✅ **Natural Language Queries**: Ask questions in plain English
- ✅ **Comprehensive Analysis**: Multiple analysis methods combined

## 🚨 Troubleshooting

### If you see dependency errors:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### If you see API key errors:
The API key is already configured in the code, but if you see errors, the system will automatically set it up.

### If you see CUDA/GPU errors:
The system will automatically fall back to CPU if GPU is not available.

## 🎉 Ready to Go!

Your Agentic RAG system is now ready for insider threat detection! The system combines:

- **Time-series processing** for CERT dataset
- **Vector embeddings** for similarity search
- **Markov Chain models** for sequence analysis
- **BERT transformers** for deep learning detection
- **LangChain agents** for natural language interaction

Start exploring with: `python quick_start.py` 