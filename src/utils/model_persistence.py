#!/usr/bin/env python3
"""
Model Persistence Utilities for Agentic RAG System
Handles saving and loading of trained models
"""

import os
import pickle
import json
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

def check_model_availability() -> Dict[str, Any]:
    """
    Check availability and validity of existing models.
    
    Returns:
        Dictionary with model status information
    """
    models_dir = "./models"
    status = {
        "markov_model": {
            "exists": False,
            "valid": False,
            "path": f"{models_dir}/markov_model.pkl",
            "last_modified": None,
            "model_info": None
        },
        "bert_model": {
            "exists": False,
            "valid": False,
            "path": f"{models_dir}/bert_model",
            "last_modified": None,
            "model_info": None
        },
        "overall_status": "train_new"
    }
    
    # Check if models directory exists
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        return status
    
    # Check Markov model
    markov_path = status["markov_model"]["path"]
    if os.path.exists(markov_path):
        status["markov_model"]["exists"] = True
        status["markov_model"]["last_modified"] = datetime.fromtimestamp(
            os.path.getmtime(markov_path)
        )
        
        # Try to load and validate Markov model
        try:
            with open(markov_path, 'rb') as f:
                markov_data = pickle.load(f)
            
            # Check if model has required attributes
            if hasattr(markov_data, 'is_fitted') and markov_data.is_fitted:
                status["markov_model"]["valid"] = True
                status["markov_model"]["model_info"] = {
                    "n_clusters": getattr(markov_data, 'n_clusters', 'unknown'),
                    "is_fitted": markov_data.is_fitted
                }
        except Exception as e:
            print(f"⚠️ Warning: Could not load Markov model: {e}")
    
    # Check BERT model directory
    bert_path = status["bert_model"]["path"]
    if os.path.exists(bert_path):
        status["bert_model"]["exists"] = True
        status["bert_model"]["last_modified"] = datetime.fromtimestamp(
            os.path.getmtime(bert_path)
        )
        
        # Check for required BERT model files
        required_files = [
            "config.json",
            "detector.pkl"
        ]
        
        bert_files_exist = all(
            os.path.exists(os.path.join(bert_path, f)) 
            for f in required_files
        )
        
        if bert_files_exist:
            try:
                # Try to load model info
                info_path = os.path.join(bert_path, "config.json")
                with open(info_path, 'r') as f:
                    bert_info = json.load(f)
                
                status["bert_model"]["valid"] = True
                status["bert_model"]["model_info"] = bert_info
            except Exception as e:
                print(f"⚠️ Warning: Could not load BERT model info: {e}")
    
    # Determine overall status
    if status["markov_model"]["exists"] and status["markov_model"]["valid"] and \
       status["bert_model"]["exists"] and status["bert_model"]["valid"]:
        status["overall_status"] = "all_existing"
    elif status["markov_model"]["exists"] or status["bert_model"]["exists"]:
        status["overall_status"] = "partial_existing"
    else:
        status["overall_status"] = "train_new"
    
    return status

def should_use_existing_models(
    force_retrain: bool = False,
    max_model_age_days: int = 30
) -> Tuple[bool, Dict[str, Any]]:
    """
    Decide whether to use existing models or train new ones.
    
    Args:
        force_retrain: If True, always force retraining.
        max_model_age_days: Retrain if models are older than this many days.
        
    Returns:
        Tuple (use_existing: bool, model_status: Dict)
    """
    status = check_model_availability()
    
    if force_retrain:
        status["overall_status"] = "force_retrain"
        return False, status
    
    if status["overall_status"] == "all_existing":
        # Check age
        now = datetime.now()
        markov_age = (now - status["markov_model"]["last_modified"]).days if status["markov_model"]["last_modified"] else 0
        bert_age = (now - status["bert_model"]["last_modified"]).days if status["bert_model"]["last_modified"] else 0
        
        if markov_age > max_model_age_days or bert_age > max_model_age_days:
            status["overall_status"] = "models_too_old"
            return False, status
        else:
            return True, status
    elif status["overall_status"] == "partial_existing":
        status["overall_status"] = "partial_existing_retrain_recommended"
        return False, status
    else:  # train_new
        return False, status

def load_existing_models(status: Dict[str, Any]) -> Tuple[Any, Any]:
    """
    Load existing Markov and BERT models based on model_status.
    
    Args:
        model_status: Dictionary from check_model_availability
        
    Returns:
        Tuple (markov_detector, bert_detector)
    """
    markov_detector = None
    bert_detector = None
    
    from src.core.markov_anomaly_detector import MarkovAnomalyDetector
    from src.core.bert_anomaly_detector import BertAnomalyDetector
    
    if status["markov_model"]["valid"]:
        try:
            with open(status["markov_model"]["path"], 'rb') as f:
                markov_detector = pickle.load(f)
            print(f"✅ Loaded Markov model from {status['markov_model']['path']}")
        except Exception as e:
            print(f"❌ Error loading Markov model: {e}")
            markov_detector = None
    
    if status["bert_model"]["valid"]:
        try:
            bert_detector = BertAnomalyDetector()
            bert_detector.load_model(status["bert_model"]["path"])
            print(f"✅ Loaded BERT model from {status['bert_model']['path']}")
        except Exception as e:
            print(f"❌ Error loading BERT model: {e}")
            bert_detector = None
            
    return markov_detector, bert_detector

def print_model_status(status: Dict[str, Any]) -> None:
    """
    Print a human-readable summary of model availability status.
    """
    print("\n📊 MODEL STATUS REPORT")
    print("=" * 50)
    
    print(f"\n🔍 MARKOV_MODEL:")
    print(f"   Exists: {'✅' if status['markov_model']['exists'] else '❌'}")
    print(f"   Valid: {'✅' if status['markov_model']['valid'] else '❌'}")
    print(f"   Path: {status['markov_model']['path']}")
    if status['markov_model']['last_modified']:
        print(f"   Last Modified: {status['markov_model']['last_modified']}")
    if status['markov_model']['model_info']:
        print(f"   Model Info: {status['markov_model']['model_info']}")
    
    print(f"\n🔍 BERT_MODEL:")
    print(f"   Exists: {'✅' if status['bert_model']['exists'] else '❌'}")
    print(f"   Valid: {'✅' if status['bert_model']['valid'] else '❌'}")
    print(f"   Path: {status['bert_model']['path']}")
    if status['bert_model']['last_modified']:
        print(f"   Last Modified: {status['bert_model']['last_modified']}")
    if status['bert_model']['model_info']:
        print(f"   Model Info: {status['bert_model']['model_info']}")
    
    print(f"\n🎯 OVERALL STATUS: {status['overall_status'].upper()}")
    print("=" * 50)

def save_trained_models(markov_detector=None, bert_detector=None, 
                       models_dir: str = "./models"):
    """Save trained models using default settings."""
    print("💾 Saving trained models...")
    
    success_count = 0
    
    if markov_detector is not None:
        try:
            model_path = os.path.join(models_dir, "markov_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(markov_detector, f)
            print(f"✅ Markov model saved to: {model_path}")
            success_count += 1
        except Exception as e:
            print(f"❌ Error saving Markov model: {e}")
    
    if bert_detector is not None:
        try:
            model_dir = os.path.join(models_dir, "bert_model")
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            # Save the detector object
            model_path = os.path.join(model_dir, "detector.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(bert_detector, f)
            
            # Save configuration
            config_path = os.path.join(model_dir, "config.json")
            config_data = {
                'model_name': bert_detector.model_name,
                'max_length': bert_detector.max_length,
                'device': str(bert_detector.device),
                'is_fitted': bert_detector.is_fitted,
                'is_trained': bert_detector.is_trained,
                'metadata': {
                    'model_type': 'bert_anomaly_detector',
                    'saved_at': datetime.now().isoformat()
                }
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            print(f"✅ BERT model saved to: {model_dir}")
            success_count += 1
        except Exception as e:
            print(f"❌ Error saving BERT model: {e}")
    
    print(f"✅ Saved {success_count} model(s) successfully!")
    return success_count 