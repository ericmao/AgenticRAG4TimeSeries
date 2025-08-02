#!/usr/bin/env python3
"""
Local LLM Integration for Agentic RAG System
Supports Gemma-2B and other local language models
"""

import os
import torch
from typing import Dict, List, Any, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.llms.base import LLM
from langchain.schema import BaseMessage, HumanMessage, AIMessage
import warnings
warnings.filterwarnings('ignore')

class LocalLLM(LLM):
    """
    Local LLM wrapper for Gemma-2B and other local models.
    Compatible with LangChain interface.
    """
    
    def __init__(self, 
                 model_name: str = "google/gemma-2b",
                 device: Optional[str] = None,
                 max_length: int = 2048,
                 temperature: float = 0.7,
                 use_4bit: bool = True,
                 use_8bit: bool = False):
        """
        Initialize local LLM.
        
        Args:
            model_name: Hugging Face model name
            device: Device to run on ('cuda', 'cpu', or None for auto)
            max_length: Maximum sequence length
            temperature: Sampling temperature
            use_4bit: Use 4-bit quantization for memory efficiency
            use_8bit: Use 8-bit quantization (alternative to 4-bit)
        """
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.temperature = temperature
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self._load_model()
        
        print(f"✅ Local LLM initialized: {model_name} on {self.device}")
    
    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            print(f"🔄 Loading {self.model_name}...")
            
            # Configure quantization
            quantization_config = None
            if self.use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            elif self.use_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            print(f"✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🔄 Falling back to CPU mode...")
            self.device = "cpu"
            self.use_4bit = False
            self.use_8bit = False
            self._load_model()
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Generate text using the local model.
        
        Args:
            prompt: Input prompt
            stop: Stop sequences
            
        Returns:
            Generated text
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=self.max_length
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    stop_token_ids=[self.tokenizer.eos_token_id] if stop else None
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return f"Error: {str(e)}"
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "local_llm"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "temperature": self.temperature
        }

class Gemma2BLLM(LocalLLM):
    """
    Specialized wrapper for Gemma-2B model.
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            model_name="google/gemma-2b",
            **kwargs
        )
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Generate text using Gemma-2B with optimized prompting.
        """
        # Format prompt for Gemma-2B
        formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=self.max_length
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    stop_token_ids=[self.tokenizer.eos_token_id] if stop else None
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            # Clean up response
            response = response.replace("<end_of_turn>", "").strip()
            
            return response
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return f"Error: {str(e)}"

def create_local_llm(model_type: str = "gemma-2b", **kwargs) -> LocalLLM:
    """
    Factory function to create local LLM instances.
    
    Args:
        model_type: Type of model ('gemma-2b', 'custom')
        **kwargs: Additional arguments for the model
        
    Returns:
        LocalLLM instance
    """
    if model_type.lower() == "gemma-2b":
        return Gemma2BLLM(**kwargs)
    else:
        return LocalLLM(**kwargs)

def test_local_llm():
    """Test the local LLM functionality."""
    print("🧪 Testing Local LLM...")
    
    try:
        # Create LLM instance
        llm = create_local_llm(
            model_type="gemma-2b",
            temperature=0.7,
            max_length=1024,
            use_4bit=True
        )
        
        # Test prompt
        test_prompt = "Explain what is anomaly detection in cybersecurity."
        print(f"📝 Test prompt: {test_prompt}")
        
        # Generate response
        response = llm._call(test_prompt)
        print(f"🤖 Response: {response}")
        
        print("✅ Local LLM test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Local LLM test failed: {e}")
        return False

if __name__ == "__main__":
    test_local_llm()
