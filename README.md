# Banking RAG System with LangChain & LangSmith

A comprehensive RAG (Retrieval-Augmented Generation) system designed for banking knowledge bases, featuring table-aware document processing and LangSmith evaluation.

## 🏗️ Architecture

- **LLM**: Mistral:latest via Ollama (local deployment)
- **Vector Store**: Chroma (local, persistent)
- **Framework**: LangChain for orchestration
- **Monitoring**: LangSmith for tracing and evaluation
- **Document Types**: Loan handbooks, regulatory manuals, policy documents

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama** and pull Mistral model:
   ```bash
   # Install Ollama from https://ollama.ai/
   ollama pull mistral:latest
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Run the System

```bash
python main.py
```

## 📊 Key Features

### 1. Table-Aware Document Processing
- Custom chunking strategy preserves table relationships
- Cross-reference resolution for "See Table X.Y" scenarios
- Metadata-rich document indexing

### 2. LangSmith Integration
- Full pipeline tracing
- Custom banking evaluators
- Real-time performance monitoring
- Cost tracking per query

### 3. Banking-Specific Optimizations
- Loan product query handling
- Regulatory compliance validation
- Rate and term accuracy checking

## 📁 Project Structure

```
├── main.py                          # Application entry point
├── config.py                        # Configuration settings
├── requirements.txt                 # Dependencies
├── Screenshot 2025-07-16 at 10.46.27 PM.png  # LangSmith dashboard screenshot
├── Screenshot 2025-07-16 at 10.46.10 PM.png  # Performance metrics screenshot
├── demo.py                          # Demo script with examples
├── src/
│   ├── banking_rag.py              # Core RAG system
│   ├── document_processors/        # Custom document loaders
│   ├── evaluation/                 # LangSmith evaluators
│   └── utils/                      # Helper utilities
├── data/
│   ├── sample_documents/           # Banking documents
│   ├── chroma_db/                  # Vector database
│   └── evaluation/                 # Test datasets
└── docs/
    ├── cost_analysis.md            # Cost optimization guide
    └── screenshots.md              # Visual documentation catalog
```

## 🔍 Evaluation Framework

The system includes comprehensive evaluation via LangSmith:

- **Banking Accuracy**: Validates numerical data (APR rates, terms)
- **Table Context**: Ensures table relationships are preserved
- **Cross-References**: Tests "See Table X" resolution
- **Compliance**: Regulatory requirement validation

## 📸 Screenshots & Examples

### LangSmith Evaluation Dashboard
![LangSmith Dashboard](Screenshot%202025-07-16%20at%2010.46.27%20PM.png)
*LangSmith evaluation dashboard showing banking RAG system performance metrics*

### System Performance Metrics  
![Performance Metrics](Screenshot%202025-07-16%20at%2010.46.10%20PM.png)
*Detailed performance and cost analysis metrics*

## 🎯 Cost Analysis

See `docs/cost_analysis.md` for detailed cost optimization strategies comparing:
- Premium setup (GPT-4 + cloud)
- Optimized setup (local models)
- Hybrid approaches

## 📈 LangSmith Monitoring

View traces and evaluations at: https://smith.langchain.com/
Project: BAT-MAN 

For visual examples of the monitoring dashboard and evaluation results, see the [Screenshots & Examples](#📸-screenshots--examples) section above.

## 🎯 Demo Commands

After running `python main.py`:

- Ask banking questions: `"What are the current mortgage rates?"`
- Run evaluation: Type `eval`
- Exit: Type `quit` 