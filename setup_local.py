#!/usr/bin/env python3
"""
Setup script for Local LLM (Gemma-2B) integration
"""

import os
import sys
import subprocess

def install_local_llm_dependencies():
    """Install dependencies for local LLM support."""
    print("📦 Installing Local LLM Dependencies...")
    
    # Core dependencies for local LLM
    local_llm_packages = [
        "accelerate>=0.20.0",
        "bitsandbytes>=0.41.0", 
        "safetensors>=0.3.0",
        "tokenizers>=0.13.0",
        "transformers>=4.20.0",
        "torch>=1.12.0"
    ]
    
    success_count = 0
    for package in local_llm_packages:
        try:
            print(f"Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                          check=True, capture_output=True)
            print(f"✅ {package} installed successfully")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
    
    print(f"✅ Installed {success_count}/{len(local_llm_packages)} packages")
    return success_count == len(local_llm_packages)

def test_local_llm_setup():
    """Test the local LLM setup."""
    print("\n🧪 Testing Local LLM Setup...")
    
    try:
        # Test imports
        import torch
        import transformers
        import accelerate
        import bitsandbytes
        
        print("✅ All required packages imported successfully")
        
        # Test GPU availability
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ GPU not available, will use CPU")
        
        # Test local LLM
        from local_llm import test_local_llm
        if test_local_llm():
            print("✅ Local LLM test passed")
            return True
        else:
            print("❌ Local LLM test failed")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Setup error: {e}")
        return False

def setup_environment():
    """Set up environment for local LLM."""
    print("🔧 Setting up Local LLM Environment...")
    
    # Set environment variables
    os.environ['LOCAL_LLM_MODEL'] = 'gemma-2b'
    os.environ['LOCAL_LLM_DEVICE'] = 'auto'
    os.environ['LOCAL_LLM_MAX_LENGTH'] = '2048'
    os.environ['LOCAL_LLM_TEMPERATURE'] = '0.1'
    os.environ['USE_4BIT_QUANTIZATION'] = 'true'
    
    print("✅ Environment variables set")
    return True

def main():
    """Main setup function."""
    print("🚀 Local LLM Setup for Agentic RAG System")
    print("=" * 50)
    
    # Setup environment
    if not setup_environment():
        print("❌ Failed to setup environment")
        return False
    
    # Install dependencies
    if not install_local_llm_dependencies():
        print("❌ Failed to install dependencies")
        return False
    
    # Test setup
    if not test_local_llm_setup():
        print("❌ Failed to test setup")
        return False
    
    print("\n🎉 Local LLM setup completed successfully!")
    print("\nYou can now run:")
    print("  python main_local.py  # Run with local LLM")
    print("  python local_llm.py   # Test local LLM only")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
