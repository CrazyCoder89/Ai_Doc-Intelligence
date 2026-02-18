# 🧠 AI Document Intelligence System (This project is just a part of an upcoming bigger project)

> **Ask questions about your documents using AI-powered Retrieval-Augmented Generation (RAG)**

A production-grade document Q&A system that runs 100% locally - no API keys, complete privacy, powered by open-source models.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

---

## ✨ Features

- 📄 **PDF Processing** - Extract and analyze text from any PDF document
- 🔍 **Semantic Search** - Find relevant information by meaning, not just keywords
- 🤖 **AI-Powered Answers** - Get accurate responses with source citations
- 🔒 **100% Local & Private** - Your documents never leave your machine
- ⚡ **Fast & Efficient** - FAISS-powered vector search in milliseconds
- 🎨 **Clean UI** - Beautiful Streamlit interface for easy interaction

---

## 🏗️ Architecture

```
PDF Upload → Text Extraction → Chunking → Embeddings → FAISS Index
                                                              ↓
User Question → Embedding → Vector Search → Retrieved Chunks → LLM → Answer
```

**Tech Stack:**
- **PDF Processing:** PyMuPDF
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store:** FAISS (Facebook AI Similarity Search)
- **LLM:** Mistral 7B via Ollama
- **Frontend:** Streamlit

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.com/download) installed

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/CrazyCoder89/Ai_Doc-Intelligence.git
cd Ai_Doc-Intelligence
```

2. **Create virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the LLM model**
```bash
ollama pull mistral
```

5. **Start Ollama server** (in a separate terminal)
```bash
ollama serve
```

6. **Run the application**
```bash
streamlit run frontend/app.py
```

7. **Open your browser** at `http://localhost:8501`

---

## 📖 Usage

1. **Upload PDFs** - Use the sidebar to upload one or more PDF documents
2. **Process Documents** - Click "Process Documents" to index them
3. **Ask Questions** - Type your question in the chat input
4. **Get Answers** - Receive AI-generated answers with source citations

### Example Questions

- "What are the main findings in this report?"
- "Summarize the financial performance in Q3"
- "What does the document say about compliance requirements?"

---

## 📁 Project Structure

```
doc_intelligence/
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
│
├── ingestion/               # PDF processing
│   ├── pdf_loader.py        # Extract text from PDFs
│   └── chunker.py           # Split text into chunks
│
├── embeddings/              # Vector conversion
│   └── embeddings.py        # Text → embeddings
│
├── retrieval/               # Search engine
│   └── vector_store.py      # FAISS operations
│
├── rag/                     # RAG pipeline
│   └── pipeline.py          # Question answering logic
│
├── frontend/                # User interface
│   └── app.py               # Streamlit application
│
└── data/                    # Data storage
    ├── raw/                 # Uploaded PDFs
    ├── processed/           # Processed chunks
    └── vector_store/        # FAISS index
```

---

## 🎯 How It Works

### 1. Document Processing
- PDFs are loaded and text is extracted page by page
- Text is split into 500-character chunks with 50-character overlap
- Each chunk is converted to a 384-dimensional vector using sentence-transformers

### 2. Indexing
- All vectors are stored in a FAISS index for fast similarity search
- Metadata (source, page number) is preserved for each chunk

### 3. Question Answering
- User's question is converted to a vector
- FAISS finds the top-5 most similar chunks
- Chunks are sent as context to the LLM (Mistral)
- LLM generates an answer grounded in the retrieved context
- Sources are cited for transparency

---

## 🛠️ Configuration

Edit `config.py` to customize:

```python
CHUNK_SIZE = 500              # Characters per chunk
CHUNK_OVERLAP = 50            # Overlap between chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # HuggingFace model
LLM_MODEL = "mistral"         # Ollama model
TOP_K_RESULTS = 5             # Number of chunks to retrieve
```

---

## 🔧 Advanced Features

### Load Existing Index
If you've already processed documents, click "Load Existing Index" in the sidebar to avoid reprocessing.

### Multi-Document Support
Upload multiple PDFs simultaneously to ask questions across all documents.

### Source Attribution
Every answer includes references to specific pages and documents for verification.

---

## 📊 Performance

- **PDF Processing:** ~5-8 seconds for a 10-page document
- **Index Building:** ~2-3 seconds for 1000 chunks
- **Query Response:** ~3-5 seconds (including LLM generation)
- **Embedding Speed:** ~1000 chunks/second on CPU

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **sentence-transformers** for excellent embedding models
- **FAISS** by Meta for blazing-fast vector search
- **Ollama** for making local LLMs accessible
- **Streamlit** for the amazing UI framework

---

## 📧 Contact

**Developer:** CrazyCoder89  
**GitHub:** [@CrazyCoder89](https://github.com/CrazyCoder89)  
**Project Link:** [https://github.com/CrazyCoder89/Ai_Doc-Intelligence](https://github.com/CrazyCoder89/Ai_Doc-Intelligence)

---

## 🌟 Star this repo if you find it useful!

Made with ❤️ using Python, AI, and open-source tools
