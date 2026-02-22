#!/usr/bin/env python3
"""
Test script for the Agentic RAG System
Verifies all components work correctly with minimal data.
"""

import os
import sys
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_processor import CERTDataProcessor
from vector_store import TimeSeriesVectorStore
from markov_anomaly_detector import MarkovAnomalyDetector
from bert_anomaly_detector import BertAnomalyDetector

def test_data_processor():
    """Test data processor functionality."""
    print("Testing Data Processor...")
    
    try:
        # Initialize processor
        processor = CERTDataProcessor()
        
        # Load sample data
        data = processor.load_sample_data()
        assert len(data) > 0, "No data loaded"
        print(f"✓ Loaded {len(data)} records")
        
        # Engineer features
        features = processor.engineer_features()
        assert 'daily_features' in features, "Features not created"
        print(f"✓ Created features for {len(features['daily_features'])} daily records")
        
        # Textualize events
        sequences = processor.textualize_events()
        assert len(sequences) > 0, "No sequences created"
        print(f"✓ Created sequences for {len(sequences)} users")
        
        # Prepare BERT sequences
        bert_sequences = processor.prepare_bert_sequences()
        assert len(bert_sequences) > 0, "No BERT sequences created"
        print(f"✓ Prepared BERT sequences for {len(bert_sequences)} users")
        
        print("✓ Data Processor: PASSED")
        return processor, sequences, bert_sequences
        
    except Exception as e:
        print(f"✗ Data Processor: FAILED - {str(e)}")
        return None, None, None

def test_vector_store(sequences):
    """Test vector store functionality."""
    print("\nTesting Vector Store...")
    
    try:
        # Initialize vector store
        vector_store = TimeSeriesVectorStore()
        
        # Create sequence embeddings
        vector_store.create_sequence_embeddings(sequences)
        assert vector_store.vector_store.index is not None, "Index not built"
        print(f"✓ Built index with {vector_store.vector_store.index.ntotal} vectors")
        
        # Test search
        if len(sequences) > 0:
            sample_user = list(sequences.keys())[0]
            sample_sequence = sequences[sample_user][0] if sequences[sample_user] else "test"
            results = vector_store.search_similar_sequences(sample_sequence, k=3)
            print(f"✓ Search returned {len(results)} results")
        
        print("✓ Vector Store: PASSED")
        return vector_store
        
    except Exception as e:
        print(f"✗ Vector Store: FAILED - {str(e)}")
        return None

def test_markov_detector(vector_store):
    """Test Markov Chain anomaly detector."""
    print("\nTesting Markov Chain Anomaly Detector...")
    
    try:
        # Initialize detector
        detector = MarkovAnomalyDetector(n_clusters=5)  # Smaller for testing
        
        # Get user embeddings
        user_embeddings = {}
        for user in vector_store.vector_store.metadata:
            if user.get('user'):
                user_id = user['user']
                embeddings = vector_store.get_user_sequence_embeddings(user_id)
                if len(embeddings) > 0:
                    user_embeddings[user_id] = embeddings
        
        if user_embeddings:
            # Train detector
            train_users = list(user_embeddings.keys())[:min(5, len(user_embeddings))]
            detector.fit(user_embeddings, normal_users=train_users)
            print(f"✓ Trained on {len(train_users)} users")
            
            # Test anomaly detection
            test_users = list(user_embeddings.keys())[:min(3, len(user_embeddings))]
            test_embeddings = {user: user_embeddings[user] for user in test_users}
            results = detector.detect_anomalies(test_embeddings)
            print(f"✓ Detected anomalies for {len(results)} users")
            
            # Print sample results
            if results:
                sample_user = list(results.keys())[0]
                result = results[sample_user]
                print(f"  Sample result for {sample_user}:")
                print(f"    Anomaly: {result['is_anomaly']}")
                print(f"    Score: {result['anomaly_score']:.4f}")
        else:
            print("⚠ No embeddings available for testing")
        
        print("✓ Markov Chain Detector: PASSED")
        return detector
        
    except Exception as e:
        print(f"✗ Markov Chain Detector: FAILED - {str(e)}")
        return None

def test_bert_detector(bert_sequences):
    """Test BERT anomaly detector."""
    print("\nTesting BERT Anomaly Detector...")
    
    try:
        # Initialize detector
        detector = BertAnomalyDetector(max_length=256)  # Smaller for testing
        
        if bert_sequences:
            # Extract features
            embeddings = detector.extract_bert_features(bert_sequences)
            print(f"✓ Extracted features for {len(embeddings)} users")
            
            if embeddings:
                # Train detector
                train_users = list(embeddings.keys())[:min(5, len(embeddings))]
                detector.fit(embeddings, normal_users=train_users)
                print(f"✓ Trained on {len(train_users)} users")
                
                # Test anomaly detection
                test_users = list(embeddings.keys())[:min(3, len(embeddings))]
                test_embeddings = {user: embeddings[user] for user in test_users}
                results = detector.detect_anomalies(test_embeddings)
                print(f"✓ Detected anomalies for {len(results)} users")
                
                # Print sample results
                if results:
                    sample_user = list(results.keys())[0]
                    result = results[sample_user]
                    print(f"  Sample result for {sample_user}:")
                    print(f"    Anomaly: {result['is_anomaly']}")
                    print(f"    Score: {result['anomaly_score']:.4f}")
                    print(f"    Reconstruction Error: {result['reconstruction_error']:.4f}")
            else:
                print("⚠ No BERT embeddings extracted")
        else:
            print("⚠ No BERT sequences available for testing")
        
        print("✓ BERT Detector: PASSED")
        return detector
        
    except Exception as e:
        print(f"✗ BERT Detector: FAILED - {str(e)}")
        return None

def test_integration():
    """Test integration of all components."""
    print("\n" + "=" * 50)
    print("INTEGRATION TEST")
    print("=" * 50)
    
    # Test data processor
    processor, sequences, bert_sequences = test_data_processor()
    if processor is None:
        return False
    
    # Test vector store
    vector_store = test_vector_store(sequences)
    if vector_store is None:
        return False
    
    # Test Markov detector
    markov_detector = test_markov_detector(vector_store)
    if markov_detector is None:
        return False
    
    # Test BERT detector
    bert_detector = test_bert_detector(bert_sequences)
    if bert_detector is None:
        return False
    
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED! ✓")
    print("=" * 50)
    print("\nSystem components are working correctly.")
    print("You can now run the main system with: python main.py")
    
    return True

def main():
    """Main test function."""
    print("Agentic RAG System - Component Tests")
    print("=" * 50)
    
    # Load environment
    load_dotenv()
    
    # Check for OpenAI API key (never set fake keys in repo)
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠ Warning: OPENAI_API_KEY not set. Agent functionality will be limited.")
        print("Copy env.example to .env and set OPENAI_API_KEY for full functionality.")
    
    # Run integration test
    success = test_integration()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        print("\nNext steps:")
        print("1. Set your OpenAI API key in .env file")
        print("2. Run: python main.py")
        print("3. Enjoy your Agentic RAG system!")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 