# 🤖 Local LLM Integration for Agentic RAG System

## Overview

This module provides local LLM (Language Model) integration for the Agentic RAG system, replacing OpenAI ChatGPT with **Google's Gemma-2B** for completely local inference.

## 🎯 Key Features

### **Local Inference**
- ✅ **No API Costs**: Completely local, no OpenAI API charges
- ✅ **Privacy**: All data stays local, no external API calls
- ✅ **Offline Capability**: Works without internet connection
- ✅ **Customizable**: Easy to switch between different local models

### **Gemma-2B Integration**
- ✅ **2B Parameters**: Efficient 2-billion parameter model
- ✅ **4-bit Quantization**: Memory-efficient loading (~4GB RAM)
- ✅ **GPU/CPU Support**: Automatic device detection and fallback
- ✅ **LangChain Compatible**: Drop-in replacement for OpenAI models

## 📦 Installation

### 1. Install Local LLM Dependencies

```bash
python setup_local.py
```

This will install:
- `accelerate>=0.20.0` - Hugging Face acceleration
- `bitsandbytes>=0.41.0` - 4-bit quantization
- `safetensors>=0.3.0` - Safe model loading
- `tokenizers>=0.13.0` - Tokenization

### 2. Test Local LLM

```bash
python local_llm.py
```

### 3. Run with Local LLM

```bash
python main_local.py
```

## 🚀 Usage

### Quick Start

```bash
# Setup local LLM
python setup_local.py

# Run the system with local LLM
python main_local.py
```

### Interactive Mode

Once running, you can ask questions like:

- `"Analyze potential insider threats for user USER0001"`
- `"Search for similar behavior patterns to unusual logon times"`
- `"What are the trends in user USER0002's activity?"`
- `"Compare Markov and BERT results for user USER0003"`

## 🔧 Configuration

### Environment Variables

Set these in your `.env` file or environment:

```env
# Local LLM Configuration
LOCAL_LLM_MODEL=gemma-2b
LOCAL_LLM_DEVICE=auto
LOCAL_LLM_MAX_LENGTH=2048
LOCAL_LLM_TEMPERATURE=0.1
USE_4BIT_QUANTIZATION=true
```

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LOCAL_LLM_MODEL` | `gemma-2b` | Model to use |
| `LOCAL_LLM_DEVICE` | `auto` | Device (auto/cuda/cpu) |
| `LOCAL_LLM_MAX_LENGTH` | `2048` | Maximum sequence length |
| `LOCAL_LLM_TEMPERATURE` | `0.1` | Sampling temperature |
| `USE_4BIT_QUANTIZATION` | `true` | Use 4-bit quantization |

## 📊 Performance

### Memory Usage
- **4-bit Quantization**: ~4GB RAM
- **8-bit Quantization**: ~8GB RAM
- **Full Precision**: ~16GB RAM

### Speed
- **CPU**: ~5-10 tokens/second
- **GPU**: ~20-50 tokens/second
- **Quality**: Good for reasoning and analysis

### Hardware Requirements

#### Minimum (CPU)
- **RAM**: 8GB
- **Storage**: 5GB for model
- **CPU**: Modern multi-core

#### Recommended (GPU)
- **RAM**: 16GB
- **GPU**: 8GB+ VRAM
- **Storage**: 10GB for model

## 🔍 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Loading  │    │  Feature        │    │  Vector Store   │
│   & Processing  │───▶│  Engineering    │───▶│  & Embeddings   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Local LLM      │    │  Markov Chain   │    │  BERT-based     │
│  (Gemma-2B)     │◀───│  Anomaly        │    │  Anomaly        │
│  Agent          │    │  Detector       │    │  Detector       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Components

### 1. **Local LLM Module** (`local_llm.py`)
- **Purpose**: Local LLM wrapper with Gemma-2B support
- **Features**:
  - 4-bit quantization for memory efficiency
  - GPU/CPU automatic detection
  - LangChain compatibility
  - Error handling and fallback

### 2. **Local Agent** (`agentic_rag_agent_local.py`)
- **Purpose**: Agentic RAG agent with local LLM
- **Features**:
  - Same 4 tools as original system
  - Local LLM integration
  - Comprehensive analysis capabilities

### 3. **Main Script** (`main_local.py`)
- **Purpose**: Main orchestration with local LLM
- **Features**:
  - Complete system initialization
  - Interactive mode
  - Example analysis

### 4. **Setup Script** (`setup_local.py`)
- **Purpose**: Automated setup for local LLM
- **Features**:
  - Dependency installation
  - Environment configuration
  - System testing

## 🔄 Migration from OpenAI

### Before (OpenAI)
```python
from langchain_openai import ChatOpenAI

import os
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.1,
    openai_api_key=os.environ.get("OPENAI_API_KEY")  # set in .env from env.example
)
```

### After (Local LLM)
```python
from local_llm import create_local_llm

llm = create_local_llm(
    model_type="gemma-2b",
    temperature=0.1,
    use_4bit=True
)
```

## 🎯 Benefits

### **Cost Savings**
- ❌ **OpenAI**: $0.002 per 1K tokens
- ✅ **Local**: $0.00 (one-time model download)

### **Privacy**
- ❌ **OpenAI**: Data sent to external servers
- ✅ **Local**: All data stays on your machine

### **Reliability**
- ❌ **OpenAI**: Depends on internet and API availability
- ✅ **Local**: Works offline, no API limits

### **Customization**
- ❌ **OpenAI**: Limited model options
- ✅ **Local**: Can use any Hugging Face model

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Use 4-bit quantization or reduce batch size

2. **Model Download Failed**
   ```
   ConnectionError: Failed to download model
   ```
   **Solution**: Check internet connection or use cached model

3. **Import Errors**
   ```
   ImportError: No module named 'bitsandbytes'
   ```
   **Solution**: Run `python setup_local.py`

4. **Slow Performance**
   ```
   Very slow token generation
   ```
   **Solution**: Use GPU if available, or reduce model size

### Performance Tips

1. **Use GPU**: Install CUDA and use GPU for faster inference
2. **4-bit Quantization**: Reduces memory usage by 75%
3. **Batch Processing**: Process multiple queries together
4. **Model Caching**: Cache model to avoid re-downloading

## 📈 Comparison

| Feature | OpenAI GPT-3.5 | Local Gemma-2B |
|---------|----------------|----------------|
| **Cost** | $0.002/1K tokens | Free |
| **Privacy** | Data sent to OpenAI | Completely local |
| **Speed** | ~100 tokens/sec | ~20 tokens/sec |
| **Memory** | N/A | ~4GB (4-bit) |
| **Quality** | Excellent | Good |
| **Reliability** | Internet dependent | Always available |

## 🎉 Getting Started

1. **Setup**:
   ```bash
   python setup_local.py
   ```

2. **Test**:
   ```bash
   python local_llm.py
   ```

3. **Run**:
   ```bash
   python main_local.py
   ```

4. **Enjoy**: Your completely local Agentic RAG system! 🚀

---

**🎯 The local LLM integration provides a cost-effective, private, and reliable alternative to OpenAI while maintaining all the functionality of your Agentic RAG system!**
