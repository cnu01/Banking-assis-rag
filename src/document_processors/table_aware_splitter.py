"""
Table-Aware Text Splitter for Banking Documents
Preserves table integrity and cross-references during chunking
"""

import re
from typing import List, Dict, Any, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import json

class TableAwareTextSplitter:
    """
    Custom text splitter that preserves table relationships and handles cross-references
    specifically designed for banking documents with complex tables.
    """
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        preserve_table_context: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_table_context = preserve_table_context
        
        # Initialize base splitter for non-table content
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Patterns for identifying tables and cross-references
        self.table_pattern = re.compile(
            r'(### Table [A-Z]?\d+\.\d+:.*?\n(?:\|.*?\|.*?\n)+)', 
            re.MULTILINE | re.DOTALL
        )
        self.cross_ref_pattern = re.compile(
            r'(see Table [A-Z]?\d+\.\d+|refer to Table [A-Z]?\d+\.\d+|Table [A-Z]?\d+\.\d+)',
            re.IGNORECASE
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents while preserving table context"""
        all_chunks = []
        
        for doc in documents:
            chunks = self._split_single_document(doc)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _split_single_document(self, document: Document) -> List[Document]:
        """Split a single document with table awareness"""
        text = document.page_content
        metadata = document.metadata.copy()
        
        # Extract tables and their positions
        tables = self._extract_tables(text)
        
        # Split text into sections around tables
        sections = self._split_around_tables(text, tables)
        
        # Process each section
        chunks = []
        for section in sections:
            if section['type'] == 'table':
                # Handle table sections specially
                table_chunks = self._process_table_section(section, metadata)
                chunks.extend(table_chunks)
            else:
                # Handle regular text sections
                text_chunks = self._process_text_section(section, metadata)
                chunks.extend(text_chunks)
        
        # Add cross-reference metadata
        chunks = self._enhance_with_cross_references(chunks)
        
        return chunks
    
    def _extract_tables(self, text: str) -> List[Dict[str, Any]]:
        """Extract table information from text"""
        tables = []
        
        for match in self.table_pattern.finditer(text):
            table_content = match.group(1)
            start_pos = match.start()
            end_pos = match.end()
            
            # Extract table identifier
            table_id_match = re.search(r'Table ([A-Z]?\d+\.\d+)', table_content)
            table_id = table_id_match.group(1) if table_id_match else "Unknown"
            
            # Extract table title
            title_match = re.search(r'### Table [A-Z]?\d+\.\d+: (.*?)\n', table_content)
            title = title_match.group(1) if title_match else "Untitled Table"
            
            tables.append({
                'id': table_id,
                'title': title,
                'content': table_content,
                'start': start_pos,
                'end': end_pos
            })
        
        return tables
    
    def _split_around_tables(self, text: str, tables: List[Dict]) -> List[Dict]:
        """Split text into sections around tables"""
        sections = []
        last_end = 0
        
        for table in tables:
            # Add text before table
            if table['start'] > last_end:
                pre_table_text = text[last_end:table['start']].strip()
                if pre_table_text:
                    sections.append({
                        'type': 'text',
                        'content': pre_table_text,
                        'context': f"Text before {table['id']}"
                    })
            
            # Add table section
            sections.append({
                'type': 'table',
                'content': table['content'],
                'table_id': table['id'],
                'table_title': table['title'],
                'context': f"Table {table['id']}: {table['title']}"
            })
            
            last_end = table['end']
        
        # Add remaining text after last table
        if last_end < len(text):
            remaining_text = text[last_end:].strip()
            if remaining_text:
                sections.append({
                    'type': 'text',
                    'content': remaining_text,
                    'context': "Text after tables"
                })
        
        return sections
    
    def _process_table_section(self, section: Dict, base_metadata: Dict) -> List[Document]:
        """Process table sections with special handling"""
        table_content = section['content']
        
        # For tables, we generally want to keep them intact
        # But if they're too large, we need to split intelligently
        if len(table_content) <= self.chunk_size:
            # Table fits in one chunk
            metadata = base_metadata.copy()
            metadata.update({
                'chunk_type': 'table',
                'table_id': section['table_id'],
                'table_title': section['table_title'],
                'context': section['context']
            })
            
            return [Document(page_content=table_content, metadata=metadata)]
        
        else:
            # Large table - split by rows while preserving headers
            return self._split_large_table(section, base_metadata)
    
    def _split_large_table(self, section: Dict, base_metadata: Dict) -> List[Document]:
        """Split large tables while preserving headers"""
        content = section['content']
        lines = content.split('\n')
        
        # Find header lines (usually first few lines after title)
        header_lines = []
        data_lines = []
        in_header = True
        
        for line in lines:
            if line.startswith('###'):  # Table title
                header_lines.append(line)
            elif line.startswith('|') and '---' in line:  # Header separator
                header_lines.append(line)
                in_header = False
            elif line.startswith('|') and in_header:  # Header row
                header_lines.append(line)
            elif line.startswith('|'):  # Data row
                data_lines.append(line)
            else:
                header_lines.append(line) if in_header else data_lines.append(line)
        
        chunks = []
        header_text = '\n'.join(header_lines)
        
        # Split data rows into chunks
        current_chunk_lines = []
        current_size = len(header_text)
        
        for line in data_lines:
            line_size = len(line) + 1  # +1 for newline
            
            if current_size + line_size > self.chunk_size and current_chunk_lines:
                # Create chunk with header + current data
                chunk_content = header_text + '\n' + '\n'.join(current_chunk_lines)
                metadata = base_metadata.copy()
                metadata.update({
                    'chunk_type': 'table_part',
                    'table_id': section['table_id'],
                    'table_title': section['table_title'],
                    'context': f"{section['context']} (Part {len(chunks) + 1})"
                })
                
                chunks.append(Document(page_content=chunk_content, metadata=metadata))
                
                current_chunk_lines = []
                current_size = len(header_text)
            
            current_chunk_lines.append(line)
            current_size += line_size
        
        # Add final chunk if there are remaining lines
        if current_chunk_lines:
            chunk_content = header_text + '\n' + '\n'.join(current_chunk_lines)
            metadata = base_metadata.copy()
            metadata.update({
                'chunk_type': 'table_part',
                'table_id': section['table_id'],
                'table_title': section['table_title'],
                'context': f"{section['context']} (Part {len(chunks) + 1})"
            })
            
            chunks.append(Document(page_content=chunk_content, metadata=metadata))
        
        return chunks
    
    def _process_text_section(self, section: Dict, base_metadata: Dict) -> List[Document]:
        """Process regular text sections"""
        # Use base splitter for text sections
        temp_doc = Document(page_content=section['content'], metadata=base_metadata)
        text_chunks = self.base_splitter.split_documents([temp_doc])
        
        # Enhance metadata
        for i, chunk in enumerate(text_chunks):
            chunk.metadata.update({
                'chunk_type': 'text',
                'context': section['context'],
                'part': i + 1 if len(text_chunks) > 1 else None
            })
        
        return text_chunks
    
    def _enhance_with_cross_references(self, chunks: List[Document]) -> List[Document]:
        """Add cross-reference information to chunk metadata"""
        # Build table reference map
        table_map = {}
        for chunk in chunks:
            if chunk.metadata.get('chunk_type') in ['table', 'table_part']:
                table_id = chunk.metadata.get('table_id')
                if table_id:
                    table_map[table_id] = chunk.metadata.get('table_title', 'Unknown Table')
        
        # Find and annotate cross-references
        for chunk in chunks:
            cross_refs = []
            for match in self.cross_ref_pattern.finditer(chunk.page_content):
                ref_text = match.group(1)
                # Extract table ID from reference
                table_id_match = re.search(r'Table ([A-Z]?\d+\.\d+)', ref_text)
                if table_id_match:
                    table_id = table_id_match.group(1)
                    if table_id in table_map:
                        cross_refs.append({
                            'referenced_table': table_id,
                            'table_title': table_map[table_id],
                            'reference_text': ref_text
                        })
            
            if cross_refs:
                chunk.metadata['cross_references'] = json.dumps(cross_refs)  # Convert to JSON string
        
        return chunks
    
    def get_table_summary(self, chunks: List[Document]) -> Dict[str, Any]:
        """Generate a summary of tables found in the chunks"""
        tables = {}
        
        for chunk in chunks:
            if chunk.metadata.get('chunk_type') in ['table', 'table_part']:
                table_id = chunk.metadata.get('table_id')
                if table_id and table_id not in tables:
                    tables[table_id] = {
                        'title': chunk.metadata.get('table_title'),
                        'parts': 0,
                        'total_content_length': 0
                    }
                
                if table_id:
                    tables[table_id]['parts'] += 1
                    tables[table_id]['total_content_length'] += len(chunk.page_content)
        
        return {
            'total_tables': len(tables),
            'tables': tables,
            'cross_references_found': sum(
                1 for chunk in chunks 
                if chunk.metadata.get('cross_references')
            )
        } 