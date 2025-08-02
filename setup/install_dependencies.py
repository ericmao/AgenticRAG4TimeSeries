#!/usr/bin/env python3
"""
Robust dependency installation script for Agentic RAG System
Handles common installation issues and provides fallback options.
"""

import subprocess
import sys
import os

def install_package(package, upgrade=False):
    """Install a single package with error handling."""
    try:
        cmd = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.append(package)
        
        print(f"Installing {package}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {package} installed successfully")
            return True
        else:
            print(f"❌ Failed to install {package}: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error installing {package}: {e}")
        return False

def install_core_dependencies():
    """Install core dependencies that are essential for the system."""
    print("📦 Installing core dependencies...")
    
    core_packages = [
        "pandas>=1.5.0",
        "numpy>=1.21.0", 
        "scikit-learn>=1.0.0",
        "python-dotenv>=0.19.0",
        "tqdm>=4.60.0"
    ]
    
    success_count = 0
    for package in core_packages:
        if install_package(package):
            success_count += 1
    
    print(f"✅ Installed {success_count}/{len(core_packages)} core packages")
    return success_count == len(core_packages)

def install_ml_dependencies():
    """Install machine learning dependencies."""
    print("\n🤖 Installing ML dependencies...")
    
    ml_packages = [
        "torch>=1.12.0",
        "transformers>=4.20.0",
        "sentence-transformers>=2.0.0",
        "faiss-cpu>=1.7.0"
    ]
    
    success_count = 0
    for package in ml_packages:
        if install_package(package):
            success_count += 1
    
    print(f"✅ Installed {success_count}/{len(ml_packages)} ML packages")
    return success_count == len(ml_packages)

def install_langchain_dependencies():
    """Install LangChain dependencies."""
    print("\n🔗 Installing LangChain dependencies...")
    
    langchain_packages = [
        "langchain>=0.0.200",
        "langchain-openai>=0.0.1",
        "openai>=1.0.0"
    ]
    
    success_count = 0
    for package in langchain_packages:
        if install_package(package):
            success_count += 1
    
    print(f"✅ Installed {success_count}/{len(langchain_packages)} LangChain packages")
    return success_count == len(langchain_packages)

def install_optional_dependencies():
    """Install optional dependencies for visualization."""
    print("\n📊 Installing optional dependencies...")
    
    optional_packages = [
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0"
    ]
    
    success_count = 0
    for package in optional_packages:
        if install_package(package):
            success_count += 1
    
    print(f"✅ Installed {success_count}/{len(optional_packages)} optional packages")
    return success_count == len(optional_packages)

def test_imports():
    """Test that key packages can be imported."""
    print("\n🧪 Testing imports...")
    
    test_imports = [
        ("pandas", "pd"),
        ("numpy", "np"),
        ("sklearn", "sklearn"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("sentence_transformers", "sentence_transformers"),
        ("faiss", "faiss"),
        ("langchain", "langchain"),
        ("openai", "openai"),
        ("dotenv", "dotenv")
    ]
    
    success_count = 0
    for module_name, import_name in test_imports:
        try:
            __import__(module_name)
            print(f"✅ {module_name} imported successfully")
            success_count += 1
        except ImportError as e:
            print(f"❌ Failed to import {module_name}: {e}")
    
    print(f"✅ {success_count}/{len(test_imports)} imports successful")
    return success_count == len(test_imports)

def main():
    """Main installation function."""
    print("🚀 Agentic RAG System - Dependency Installation")
    print("=" * 50)
    
    # Upgrade pip first
    print("📦 Upgrading pip...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      capture_output=True)
        print("✅ Pip upgraded successfully")
    except Exception as e:
        print(f"⚠ Warning: Could not upgrade pip: {e}")
    
    # Install dependencies in stages
    core_success = install_core_dependencies()
    ml_success = install_ml_dependencies()
    langchain_success = install_langchain_dependencies()
    optional_success = install_optional_dependencies()
    
    # Test imports
    import_success = test_imports()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 INSTALLATION SUMMARY")
    print("=" * 50)
    print(f"Core Dependencies: {'✅' if core_success else '❌'}")
    print(f"ML Dependencies: {'✅' if ml_success else '❌'}")
    print(f"LangChain Dependencies: {'✅' if langchain_success else '❌'}")
    print(f"Optional Dependencies: {'✅' if optional_success else '❌'}")
    print(f"Import Tests: {'✅' if import_success else '❌'}")
    
    if core_success and ml_success and langchain_success and import_success:
        print("\n🎉 All essential dependencies installed successfully!")
        print("You can now run: python main.py")
        return True
    else:
        print("\n⚠ Some dependencies failed to install.")
        print("The system may work with limited functionality.")
        print("Try running: python test_system.py")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 