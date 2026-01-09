# Agentic RAG System

Industry-grade Retrieval-Augmented Generation system with intelligent query routing, guardrails, and multi-source information retrieval.

## Features

- **Multi-document Upload**: PDF, DOCX, TXT support
- **Intelligent Routing**: Automatically routes queries to:
  - RAG (uploaded documents)
  - LLM knowledge
  - Internet search (Serper API)
- **Guardrails**:
  - Input validation (malicious query detection)
  - Relevance scoring
  - Hallucination detection
- **Source Attribution**: Clear citations for all responses
- **Streamlit UI**: User-friendly interface
- **Evaluation Framework**: Comprehensive logging and metrics

## Setup

### 1. Prerequisites

- Python 3.9+
- Groq API key ([Get it here](https://console.groq.com/))
- Serper API key ([Get it here](https://serper.dev/))

### 2. Installation

```bash
# Navigate to project directory
cd C:\Users\ashfa\OneDrive\Desktop\My-Learning\Test

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
SERPER_API_KEY=your_serper_api_key_here
```

You can copy `.env.example` and fill in your API keys:

```bash
copy .env.example .env
```

### 4. Run Application

```bash
streamlit run main.py
```

The application will open in your browser at `http://localhost:8501`.

## Usage

1. **Upload Documents**: Click "Browse files" to upload PDF, DOCX, or TXT files
2. **Process Documents**: Click "Process Documents" to index them
3. **Ask Questions**: Type your question in the chat input
4. **View Sources**: Expand the "Sources" section to see where information came from

## Architecture

```
Query → Input Validation → Query Analysis → Router
                                             ├─→ RAG Tool → Relevance Check → Response Synthesis
                                             ├─→ LLM Tool → Response
                                             └─→ Search Tool → Response
                                                              ↓
                                            Hallucination Check → Final Response
```

## Configuration

Edit `.env` or `config.py` to customize:

- Chunk size and overlap
- Top-K retrieval
- Similarity thresholds
- Relevance/hallucination thresholds
- LLM model selection

## Evaluation

View metrics by clicking "View Metrics" in the sidebar:

- Response times
- Source distribution
- Relevance scores
- Error rates

Logs are stored in `./logs/`.

## Project Structure

```
Test/
├── data/                    # Data persistence
│   ├── uploaded_docs/      # Original documents
│   └── chroma_db/          # Vector database
├── logs/                    # Application logs
├── src/                     # Source code
│   ├── agents/             # Agentic workflow (LangGraph)
│   ├── document_processing/ # Document loaders and chunking
│   ├── evaluation/         # Logging and metrics
│   ├── guardrails/         # Safety and quality checks
│   ├── llm/                # Groq client
│   ├── search/             # Serper client
│   ├── ui/                 # Streamlit components
│   └── vector_store/       # ChromaDB management
├── tests/                   # Unit tests
├── config.py               # Configuration management
├── main.py                 # Application entry point
└── requirements.txt        # Dependencies
```

## Troubleshooting

**Issue**: "No module named 'src'"
- **Solution**: Ensure you're running from the project root directory

**Issue**: "API key not found"
- **Solution**: Check `.env` file exists and contains valid API keys

**Issue**: "ChromaDB error"
- **Solution**: Delete `./data/chroma_db/` and restart

## License

MIT
