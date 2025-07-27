#!/usr/bin/env python3
"""
Demo script for Agentic RAG System
Shows various types of queries and their results
"""

import os
import sys
from main import initialize_components, load_environment

def run_demo_queries():
    """Run a series of demo queries to showcase the system capabilities."""
    
    print("🎯 AGENTIC RAG SYSTEM DEMO")
    print("=" * 60)
    
    # Load environment and initialize components
    config = load_environment()
    data_processor, vector_store, markov_detector, bert_detector, agent = initialize_components(config)
    
    # Demo queries
    demo_queries = [
        "USER0001",
        "分析 USER0005 是否有異常行為",
        "late-night login patterns",
        "suspicious user behavior",
        "USER0010",
        "unusual login times",
        "analyze USER0003 for anomalies"
    ]
    
    print("\n📋 DEMO QUERIES:")
    print("=" * 60)
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        print("-" * 40)
        
        try:
            result = agent.agent_executor.invoke({"input": query})
            print(f"✅ Result: {result['output']}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("-" * 40)
    
    # Show system capabilities
    print("\n📊 SYSTEM CAPABILITIES SUMMARY:")
    print("=" * 60)
    
    status = agent.get_system_status()
    for component, info in status.items():
        print(f"\n{component.upper()}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    print("\n🎉 Demo completed successfully!")
    print("The system demonstrates:")
    print("✅ Natural language query processing")
    print("✅ Multi-language support (English/Chinese)")
    print("✅ Time-series analysis")
    print("✅ Anomaly detection (Markov + BERT)")
    print("✅ Vector similarity search")
    print("✅ Agentic reasoning and tool selection")

if __name__ == "__main__":
    run_demo_queries() 