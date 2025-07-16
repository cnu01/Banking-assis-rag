"""
Banking RAG System - Core Implementation
Integrates LangChain components with table-aware processing and LangSmith tracing
"""

import os
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks import LangChainTracer
from langchain.schema import Document
from langsmith import Client

import config
from src.document_processors.banking_document_loader import BankingDocumentLoader


class BankingRAGSystem:
    """
    Complete Banking RAG System with LangSmith integration
    Features table-aware chunking, cross-reference resolution, and banking-specific prompts
    """
    
    def __init__(self):
        # Initialize LangSmith tracing
        self.langsmith_client = Client()
        self.tracer = LangChainTracer(client=self.langsmith_client)
        
        # Initialize components
        self.document_loader = None
        self.vectorstore = None
        self.embeddings = None
        self.llm = None
        self.qa_chain = None
        self.conversation_chain = None
        self.memory = None
        
        # Document cache
        self.documents = []
        self.document_summary = {}
        
        print("🏦 Banking RAG System initialized with LangSmith tracing")
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all RAG components"""
        
        # 1. Initialize embeddings (local HuggingFace model)
        print("📊 Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 2. Initialize Ollama LLM (Mistral)
        print("🤖 Connecting to Ollama (Mistral)...")
        self.llm = Ollama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            callbacks=[self.tracer]
        )
        
        # 3. Initialize document loader
        self.document_loader = BankingDocumentLoader(config.DOCUMENTS_DIR)
        
        # 4. Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        print("✅ Components initialized successfully")
    
    def load_documents(self):
        """Load and process banking documents"""
        print("📄 Loading banking documents...")
        
        # Load documents with table-aware processing
        self.documents = self.document_loader.load_all_documents()
        
        # Generate document summary
        self.document_summary = self.document_loader.get_document_summary(self.documents)
        
        # Create vector store
        print("🔍 Creating vector store...")
        
        # Filter complex metadata for Chroma compatibility
        filtered_documents = filter_complex_metadata(self.documents)
        
        self.vectorstore = Chroma.from_documents(
            documents=filtered_documents,
            embedding=self.embeddings,
            persist_directory=config.CHROMA_PERSIST_DIRECTORY
        )
        
        # Initialize retrieval chains
        self._initialize_chains()
        
        # Print summary
        self._print_loading_summary()
    
    def _initialize_chains(self):
        """Initialize LangChain retrieval chains with banking-specific prompts"""
        
        # Banking-specific prompt template
        banking_prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template="""You are a knowledgeable banking assistant with expertise in loan products, regulatory requirements, and banking policies. 

Use the following context to answer the question. Pay special attention to:
- Exact numerical values (APR rates, loan amounts, terms)
- Table relationships and cross-references
- Regulatory compliance requirements
- Current effective dates of rates and policies

Context from banking documents:
{context}

Previous conversation:
{chat_history}

Question: {question}

Important Instructions:
1. If the question involves specific rates or numerical data, cite the exact table or section
2. For cross-references (e.g., "see Table X.Y"), provide the referenced information
3. If information is not in the context, clearly state "This information is not available in the current documents"
4. For regulatory questions, emphasize compliance requirements and deadlines
5. Always mention if rates or information are subject to change or approval

Answer:"""
        )
        
        # Create retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": config.TOP_K_RETRIEVAL}
            ),
            chain_type_kwargs={
                "prompt": banking_prompt
            },
            callbacks=[self.tracer]
        )
        
        # Create conversational chain for context awareness
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": config.TOP_K_RETRIEVAL}
            ),
            memory=self.memory,
            callbacks=[self.tracer],
            verbose=True
        )
    
    def ask_question(self, question: str, use_conversation: bool = True) -> str:
        """
        Ask a question to the banking RAG system
        
        Args:
            question: The banking-related question
            use_conversation: Whether to use conversation history
            
        Returns:
            The answer from the RAG system
        """
        
        try:
            # Enhance question with banking context if needed
            enhanced_question = self._enhance_question(question)
            
            # Get answer from appropriate chain
            if use_conversation and self.conversation_chain:
                result = self.conversation_chain({"question": enhanced_question})
                answer = result["answer"]
            else:
                result = self.qa_chain({"query": enhanced_question})
                answer = result["result"]
            
            # Post-process answer for banking compliance
            processed_answer = self._post_process_answer(answer, question)
            
            return processed_answer
            
        except Exception as e:
            raise e
    
    def _enhance_question(self, question: str) -> str:
        """Enhance question with banking context clues"""
        
        # Check for specific banking terms and add context
        banking_terms = {
            'rate': 'current interest rate',
            'apr': 'Annual Percentage Rate',
            'mortgage': 'home loan mortgage',
            'loan': 'lending product',
            'compliance': 'regulatory compliance requirement',
            'fdic': 'FDIC regulatory requirement'
        }
        
        enhanced = question.lower()
        for term, context in banking_terms.items():
            if term in enhanced and context not in enhanced:
                question = f"{question} (related to {context})"
                break
        
        return question
    
    def _post_process_answer(self, answer: str, original_question: str) -> str:
        """Post-process answer to ensure banking compliance and accuracy"""
        
        # Add disclaimers for rate-related questions
        if any(term in original_question.lower() for term in ['rate', 'apr', 'cost', 'fee']):
            if "subject to change" not in answer.lower():
                answer += "\n\n⚠️ Note: All rates and fees are subject to change and credit approval."
        
        # Add compliance note for regulatory questions
        if any(term in original_question.lower() for term in ['compliance', 'regulation', 'requirement']):
            if "consult" not in answer.lower():
                answer += "\n\n📋 For specific compliance guidance, consult with the compliance department."
        
        return answer
    
    def search_tables(self, query: str) -> List[Document]:
        """Search specifically for table content"""
        
        # Filter to table documents
        table_docs = self.document_loader.get_documents_with_tables(self.documents)
        
        if not table_docs:
            return []
        
        # Create temporary vector store with only table content
        table_vectorstore = Chroma.from_documents(
            documents=table_docs,
            embedding=self.embeddings
        )
        
        # Search tables
        results = table_vectorstore.similarity_search(query, k=5)
        
        return results
    
    def resolve_cross_reference(self, table_reference: str) -> Optional[Document]:
        """Resolve a cross-reference to a specific table"""
        
        # Find documents with the referenced table
        for doc in self.documents:
            table_id = doc.metadata.get('table_id')
            if table_id and table_reference in table_id:
                return doc
        
        return None
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics and health information"""
        
        stats = {
            'documents_loaded': len(self.documents),
            'vectorstore_ready': self.vectorstore is not None,
            'chains_ready': self.qa_chain is not None,
            'memory_conversations': len(self.memory.chat_memory.messages) if self.memory else 0,
            'document_summary': self.document_summary,
            'langsmith_project': config.LANGSMITH_PROJECT
        }
        
        return stats
    
    def _print_loading_summary(self):
        """Print a summary of loaded documents"""
        summary = self.document_summary
        
        print("\n" + "="*50)
        print("📊 BANKING RAG SYSTEM - LOADING SUMMARY")
        print("="*50)
        
        print(f"📄 Total chunks: {summary['total_chunks']}")
        print(f"📁 Files loaded: {summary['files_loaded']}")
        
        print("\n🏷️ Document Types:")
        for doc_type, count in summary['document_types'].items():
            print(f"  • {doc_type}: {count} chunks")
        
        print("\n📊 Content Types:")
        for content_type, count in summary['content_types'].items():
            print(f"  • {content_type}: {count} chunks")
        
        table_summary = summary['table_summary']
        print(f"\n📋 Tables Found: {table_summary['total_tables']}")
        for table_id, info in table_summary['tables'].items():
            print(f"  • Table {table_id}: {info['title']} ({info['parts']} parts)")
        
        print(f"\n🔗 Cross-references: {table_summary['cross_references_found']} chunks")
        print(f"💹 Numerical data: {summary['chunks_with_numerical_data']} chunks")
        
        print("\n✅ RAG System ready for questions!")
        print("="*50)
    
    def test_table_preservation(self):
        """Test that table relationships are preserved"""
        
        # Test queries that should reference tables
        test_queries = [
            "What are the personal loan rates?",
            "Show me the FDIC capital requirements",
            "What information is in Table 1.1?",
            "Tell me about mortgage rates"
        ]
        
        print("\n🧪 Testing table preservation...")
        
        for query in test_queries:
            print(f"\n❓ Query: {query}")
            try:
                answer = self.ask_question(query, use_conversation=False)
                print(f"💡 Answer preview: {answer[:100]}...")
                
                # Check if answer contains table references
                if any(word in answer.lower() for word in ['table', 'rate', '%']):
                    print("✅ Table content detected in answer")
                else:
                    print("⚠️ No clear table content in answer")
                    
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        print("\n🧪 Table preservation test completed") 