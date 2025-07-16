"""
Banking RAG System Demo Script
Demonstrates key features including table-aware processing and LangSmith evaluation

📸 For visual examples of LangSmith evaluation results and system performance,
   see the screenshots referenced in README.md and docs/cost_analysis.md
"""

import config  # Load configuration and set environment variables
from src.banking_rag import BankingRAGSystem
from src.evaluation.langsmith_evaluator import LangSmithEvaluator
import time

def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"🏦 {title}")
    print("="*60)

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*20} {title} {'='*20}")

def demo_system_initialization():
    """Demonstrate system initialization"""
    print_header("BANKING RAG SYSTEM DEMO")
    print("🚀 Initializing Banking RAG System with LangSmith Tracing...")
    
    try:
        rag_system = BankingRAGSystem()
        print("✅ System initialized successfully!")
        return rag_system
    except Exception as e:
        print(f"❌ Initialization failed: {str(e)}")
        print("💡 Make sure Ollama is running with Mistral model")
        print("   Run: ollama pull mistral:latest")
        return None

def demo_document_loading(rag_system: BankingRAGSystem):
    """Demonstrate document loading and processing"""
    print_section("DOCUMENT LOADING & PROCESSING")
    
    try:
        rag_system.load_documents()
        
        # Show system stats
        stats = rag_system.get_system_stats()
        print(f"\n📊 System Statistics:")
        print(f"   Documents loaded: {stats['documents_loaded']}")
        print(f"   Vector store ready: {stats['vectorstore_ready']}")
        print(f"   LangSmith project: {stats['langsmith_project']}")
        
        return True
    except Exception as e:
        print(f"❌ Document loading failed: {str(e)}")
        return False

def demo_basic_queries(rag_system: BankingRAGSystem):
    """Demonstrate basic banking queries"""
    print_section("BASIC BANKING QUERIES")
    
    basic_queries = [
        "What are the current personal loan rates?",
        "Show me the FHA mortgage rates",
        "What are the FDIC capital requirements?",
        "Tell me about the CD rates for 12 months"
    ]
    
    for i, query in enumerate(basic_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        try:
            start_time = time.time()
            answer = rag_system.ask_question(query, use_conversation=False)
            end_time = time.time()
            
            print(f"💡 Answer: {answer[:200]}...")
            print(f"⏱️  Response time: {end_time - start_time:.2f}s")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def demo_table_preservation(rag_system: BankingRAGSystem):
    """Demonstrate table-aware processing"""
    print_section("TABLE-AWARE PROCESSING")
    
    table_queries = [
        "What information is in Table 1.1?",
        "Show me the amortization calculation from Table 4.1",
        "Compare rates in Table 2.1 and Table 2.2",
        "What are the rate adjustments in Table M.4?"
    ]
    
    print("🧪 Testing table relationship preservation...")
    
    for i, query in enumerate(table_queries, 1):
        print(f"\n📋 Table Query {i}: {query}")
        try:
            answer = rag_system.ask_question(query, use_conversation=False)
            
            # Check if table content is preserved
            has_table_ref = any(word in answer for word in ['Table', 'table'])
            has_numerical = any(char in answer for char in ['%', '$'])
            has_structure = '|' in answer or '\n' in answer
            
            print(f"💡 Answer: {answer[:150]}...")
            print(f"✅ Checks: Table ref: {has_table_ref}, Numbers: {has_numerical}, Structure: {has_structure}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def demo_cross_references(rag_system: BankingRAGSystem):
    """Demonstrate cross-reference resolution"""
    print_section("CROSS-REFERENCE RESOLUTION")
    
    cross_ref_queries = [
        "The document mentions see Table 1.1 - what does it contain?",
        "What is the Table 4.1 referenced for amortization?",
        "Show me details about Table 2.2 mentioned in the mortgage section"
    ]
    
    for i, query in enumerate(cross_ref_queries, 1):
        print(f"\n🔗 Cross-ref Query {i}: {query}")
        try:
            answer = rag_system.ask_question(query, use_conversation=False)
            print(f"💡 Answer: {answer[:200]}...")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def demo_conversation_context(rag_system: BankingRAGSystem):
    """Demonstrate conversation memory"""
    print_section("CONVERSATION CONTEXT")
    
    conversation_flow = [
        "What are personal loan rates?",
        "What about for excellent credit specifically?",
        "How does that compare to auto loan rates?",
        "What would be my monthly payment for $25,000?"
    ]
    
    print("🗣️  Testing conversation context preservation...")
    
    for i, query in enumerate(conversation_flow, 1):
        print(f"\n💬 Turn {i}: {query}")
        try:
            answer = rag_system.ask_question(query, use_conversation=True)
            print(f"💡 Answer: {answer[:150]}...")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def demo_compliance_features(rag_system: BankingRAGSystem):
    """Demonstrate banking compliance features"""
    print_section("BANKING COMPLIANCE FEATURES")
    
    compliance_queries = [
        "What are the BSA reporting requirements?",
        "Tell me about TILA disclosure requirements",
        "What happens if a bank becomes undercapitalized?",
        "When do I need to file a SAR?"
    ]
    
    for i, query in enumerate(compliance_queries, 1):
        print(f"\n⚖️  Compliance Query {i}: {query}")
        try:
            answer = rag_system.ask_question(query, use_conversation=False)
            
            # Check for compliance features
            has_disclaimer = any(phrase in answer.lower() for phrase in 
                               ['subject to', 'approval', 'consult', 'compliance'])
            has_regulation = any(reg in answer.upper() for reg in 
                               ['FDIC', 'BSA', 'TILA', 'SAR', 'CTR'])
            
            print(f"💡 Answer: {answer[:150]}...")
            print(f"✅ Compliance: Disclaimer: {has_disclaimer}, Regulation: {has_regulation}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def demo_langsmith_evaluation(rag_system: BankingRAGSystem):
    """Demonstrate LangSmith evaluation framework"""
    print_section("LANGSMITH EVALUATION")
    
    print("🔍 Running comprehensive banking RAG evaluation...")
    print("📊 This will test accuracy, table context, compliance, and cross-references")
    
    try:
        evaluator = LangSmithEvaluator(rag_system)
        print("\n⚡ Starting evaluation (this may take a few minutes)...")
        
        # Run a subset of evaluations for demo
        demo_datasets = {
            'loan_products': evaluator.test_datasets['loan_products'][:2],
            'regulatory_compliance': evaluator.test_datasets['regulatory_compliance'][:2],
            'table_cross_references': evaluator.test_datasets['table_cross_references'][:2]
        }
        
        evaluator.test_datasets = demo_datasets
        results = evaluator.run_evaluation()
        
        print("\n🎯 Evaluation completed!")
        print("📈 View detailed traces at: https://smith.langchain.com/")
        print(f"📋 Project: {config.LANGSMITH_PROJECT}")
        
    except Exception as e:
        print(f"❌ Evaluation error: {str(e)}")
        print("💡 Check LangSmith configuration in config.py")

def demo_system_stats(rag_system: BankingRAGSystem):
    """Show final system statistics"""
    print_section("SYSTEM STATISTICS")
    
    stats = rag_system.get_system_stats()
    summary = rag_system.document_summary
    
    print("📊 Final System Status:")
    print(f"   ✅ Documents processed: {stats['documents_loaded']}")
    print(f"   ✅ Vector store ready: {stats['vectorstore_ready']}")
    print(f"   ✅ Chains initialized: {stats['chains_ready']}")
    print(f"   ✅ Conversation turns: {stats['memory_conversations']}")
    
    print(f"\n📋 Document Analysis:")
    print(f"   📁 Files loaded: {summary['files_loaded']}")
    print(f"   📄 Total chunks: {summary['total_chunks']}")
    print(f"   🔢 Numerical data chunks: {summary['chunks_with_numerical_data']}")
    
    table_summary = summary['table_summary']
    print(f"\n📊 Table Processing:")
    print(f"   📋 Tables found: {table_summary['total_tables']}")
    print(f"   🔗 Cross-references: {table_summary['cross_references_found']}")
    
    print(f"\n🔍 Content Types:")
    for content_type, count in summary['content_types'].items():
        print(f"   • {content_type}: {count} chunks")

def main():
    """Main demo execution"""
    print_header("BANKING RAG SYSTEM - COMPREHENSIVE DEMO")
    print("This demo showcases:")
    print("• Table-aware document processing")
    print("• Cross-reference resolution")
    print("• Banking compliance features")
    print("• LangSmith evaluation framework")
    print("• Conversation context management")
    
    # Initialize system
    rag_system = demo_system_initialization()
    if not rag_system:
        return
    
    # Load documents
    if not demo_document_loading(rag_system):
        return
    
    # Run demonstrations
    demo_basic_queries(rag_system)
    demo_table_preservation(rag_system)
    demo_cross_references(rag_system)
    demo_conversation_context(rag_system)
    demo_compliance_features(rag_system)
    
    # Optional: Run evaluation (comment out if too slow)
    print("\n❓ Run LangSmith evaluation? (y/n): ", end="")
    if input().lower().startswith('y'):
        demo_langsmith_evaluation(rag_system)
    
    # Show final stats
    demo_system_stats(rag_system)
    
    print_header("DEMO COMPLETED")
    print("🎉 Banking RAG System demonstration completed successfully!")
    print("\n📚 Key Features Demonstrated:")
    print("   ✅ Table relationship preservation")
    print("   ✅ Cross-reference resolution")
    print("   ✅ Banking compliance integration")
    print("   ✅ LangSmith tracing and evaluation")
    print("   ✅ Conversation context management")
    print("   ✅ Cost-optimized local deployment")
    
    print("\n🔗 Next Steps:")
    print("   • View traces: https://smith.langchain.com/")
    print("   • Read cost analysis: docs/cost_analysis.md")
    print("   • Run full evaluation: python -c 'from main import *; main()'")
    print("   • Customize for your banking documents")

if __name__ == "__main__":
    main() 