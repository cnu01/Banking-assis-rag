"""
Banking RAG System - Main Application
Demonstrates LangChain-based RAG with table-aware chunking and LangSmith tracing
"""

import config  # Load configuration and set environment variables
from src.banking_rag import BankingRAGSystem
from src.evaluation.langsmith_evaluator import LangSmithEvaluator

def main():
    """Main application entry point"""
    print("🏦 Banking RAG System with LangSmith Tracing")
    print("=" * 50)
    
    # Initialize the RAG system
    print("📊 Initializing Banking RAG System...")
    rag_system = BankingRAGSystem()
    
    # Load and process documents
    print("📄 Loading sample banking documents...")
    rag_system.load_documents()
    
    # Interactive demo
    print("\n🤖 RAG System Ready! Ask questions about banking products.")
    print("Type 'quit' to exit, 'eval' to run evaluation")
    print("-" * 50)
    
    while True:
        question = input("\n❓ Your question: ").strip()
        
        if question.lower() == 'quit':
            break
        elif question.lower() == 'eval':
            print("🔍 Running LangSmith evaluation...")
            evaluator = LangSmithEvaluator(rag_system)
            evaluator.run_evaluation()
        elif question:
            try:
                answer = rag_system.ask_question(question)
                print(f"\n💡 Answer: {answer}")
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
    
    print("\n👋 Goodbye!")

if __name__ == "__main__":
    main() 