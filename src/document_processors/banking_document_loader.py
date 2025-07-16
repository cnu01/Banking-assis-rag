"""
Banking Document Loader
Specialized loader for banking documents with enhanced metadata extraction
"""

import os
from typing import List, Dict, Any
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.schema import Document
from .table_aware_splitter import TableAwareTextSplitter
import json


class BankingDocumentLoader:
    """
    Specialized document loader for banking documents that preserves
    table relationships and extracts banking-specific metadata.
    """
    
    def __init__(self, documents_dir: str):
        self.documents_dir = Path(documents_dir)
        self.table_splitter = TableAwareTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Banking document type mapping
        self.doc_type_mapping = {
            'loan_handbook': {
                'type': 'loan_products',
                'contains': ['rates', 'terms', 'amortization'],
                'priority': 'high'
            },
            'regulatory_manual': {
                'type': 'compliance',
                'contains': ['regulations', 'requirements', 'procedures'],
                'priority': 'critical'
            },
            'rate_sheet': {
                'type': 'current_rates',
                'contains': ['rates', 'pricing', 'fees'],
                'priority': 'high'
            },
            'policy': {
                'type': 'internal_policy',
                'contains': ['procedures', 'guidelines', 'standards'],
                'priority': 'medium'
            }
        }
    
    def load_all_documents(self) -> List[Document]:
        """Load all banking documents from the documents directory"""
        all_documents = []
        
        if not self.documents_dir.exists():
            raise FileNotFoundError(f"Documents directory not found: {self.documents_dir}")
        
        # Load all text and PDF files
        for file_path in self.documents_dir.glob("*"):
            if file_path.suffix.lower() in ['.txt', '.pdf']:
                try:
                    documents = self._load_single_file(file_path)
                    all_documents.extend(documents)
                    print(f"✅ Loaded: {file_path.name} ({len(documents)} chunks)")
                except Exception as e:
                    print(f"❌ Error loading {file_path.name}: {str(e)}")
        
        print(f"\n📊 Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def _load_single_file(self, file_path: Path) -> List[Document]:
        """Load and process a single document file"""
        
        # Determine document type from filename
        doc_type_info = self._identify_document_type(file_path.name)
        
        # Load raw document
        if file_path.suffix.lower() == '.pdf':
            loader = PyPDFLoader(str(file_path))
        else:
            loader = TextLoader(str(file_path), encoding='utf-8')
        
        raw_documents = loader.load()
        
        # Add banking-specific metadata
        for doc in raw_documents:
            doc.metadata.update({
                'source_file': file_path.name,
                'file_path': str(file_path),
                'document_type': doc_type_info['type'],
                'contains_topics': ', '.join(doc_type_info['contains']),  # Convert list to string
                'priority_level': doc_type_info['priority'],
                'load_timestamp': self._get_timestamp()
            })
        
        # Split documents with table awareness
        chunked_documents = self.table_splitter.split_documents(raw_documents)
        
        # Add chunk-specific banking metadata
        self._enhance_banking_metadata(chunked_documents)
        
        return chunked_documents
    
    def _identify_document_type(self, filename: str) -> Dict[str, Any]:
        """Identify document type based on filename"""
        filename_lower = filename.lower()
        
        for key, info in self.doc_type_mapping.items():
            if key in filename_lower:
                return info
        
        # Default for unknown document types
        return {
            'type': 'general_banking',
            'contains': ['general'],
            'priority': 'low'
        }
    
    def _enhance_banking_metadata(self, documents: List[Document]) -> None:
        """Add banking-specific metadata to document chunks"""
        
        for doc in documents:
            content = doc.page_content.lower()
            
            # Detect banking-specific content types
            content_types = []
            
            # Loan products
            if any(term in content for term in ['apr', 'loan', 'mortgage', 'rate', 'payment']):
                content_types.append('loan_products')
            
            # Rates and pricing
            if any(term in content for term in ['%', 'rate', 'apr', 'fee', 'cost', 'price']):
                content_types.append('rates_pricing')
            
            # Regulatory content
            if any(term in content for term in ['compliance', 'regulation', 'fdic', 'requirement']):
                content_types.append('regulatory')
            
            # Table content
            if doc.metadata.get('chunk_type') in ['table', 'table_part']:
                content_types.append('tabular_data')
            
            # Amortization/calculations
            if any(term in content for term in ['amortization', 'payment', 'principal', 'interest']):
                content_types.append('calculations')
            
            doc.metadata['content_types'] = ', '.join(content_types)  # Convert list to string
            
            # Extract numerical data for banking accuracy checks
            self._extract_numerical_data(doc)
    
    def _extract_numerical_data(self, document: Document) -> None:
        """Extract numerical data (rates, amounts, etc.) for validation"""
        import re
        
        content = document.page_content
        numerical_data = {}
        
        # Extract percentage rates
        rate_pattern = r'(\d+\.?\d*)\s*%'
        rates = re.findall(rate_pattern, content)
        if rates:
            numerical_data['rates'] = [float(rate) for rate in rates]
        
        # Extract dollar amounts
        dollar_pattern = r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        amounts = re.findall(dollar_pattern, content)
        if amounts:
            numerical_data['dollar_amounts'] = [
                float(amount.replace(',', '')) for amount in amounts
            ]
        
        # Extract loan terms (months/years)
        term_pattern = r'(\d+)\s*(?:month|year)s?'
        terms = re.findall(term_pattern, content, re.IGNORECASE)
        if terms:
            numerical_data['terms'] = [int(term) for term in terms]
        
        if numerical_data:
            document.metadata['numerical_data'] = json.dumps(numerical_data)  # Convert to JSON string
    
    def get_document_summary(self, documents: List[Document]) -> Dict[str, Any]:
        """Generate a summary of loaded documents"""
        
        # Count by document type
        type_counts = {}
        content_type_counts = {}
        total_chunks = len(documents)
        
        for doc in documents:
            doc_type = doc.metadata.get('document_type', 'unknown')
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            
            content_types = doc.metadata.get('content_types', '').split(', ') # Convert comma-separated string back to list
            for ct in content_types:
                if ct.strip():  # Only count non-empty content types
                    content_type_counts[ct] = content_type_counts.get(ct, 0) + 1
        
        # Get table summary
        table_summary = self.table_splitter.get_table_summary(documents)
        
        return {
            'total_chunks': total_chunks,
            'document_types': type_counts,
            'content_types': content_type_counts,
            'table_summary': table_summary,
            'files_loaded': len(set(doc.metadata.get('source_file') for doc in documents)),
            'chunks_with_numerical_data': sum(
                1 for doc in documents 
                if doc.metadata.get('numerical_data')
            )
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_documents_by_type(self, documents: List[Document], doc_type: str) -> List[Document]:
        """Filter documents by type"""
        return [
            doc for doc in documents 
            if doc.metadata.get('document_type') == doc_type
        ]
    
    def get_documents_with_tables(self, documents: List[Document]) -> List[Document]:
        """Get all documents that contain tables"""
        return [
            doc for doc in documents 
            if doc.metadata.get('chunk_type') in ['table', 'table_part']
        ]
    
    def get_cross_referenced_documents(self, documents: List[Document]) -> List[Document]:
        """Get documents that contain cross-references"""
        return [
            doc for doc in documents 
            if doc.metadata.get('cross_references')
        ] 