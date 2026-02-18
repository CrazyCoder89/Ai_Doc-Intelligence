# frontend/app.py
# This is the Streamlit web interface for our RAG system.
# Users can upload PDFs, ask questions, and get answers with sources.

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from pathlib import Path
import shutil

# Import our backend components
from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_pages
from embeddings.embeddings import embed_chunks, embed_text
from retrieval.vector_store import VectorStore
from rag.pipeline import generate_answer
from config import RAW_DIR, VECTOR_STORE_DIR

# Page config
st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="📄",
    layout="wide"
)

# Title
st.title("📄 AI Document Intelligence System")
st.markdown("*Ask questions about your documents with AI-powered RAG*")
st.divider()

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

# Sidebar for file upload
with st.sidebar:
    st.header("📁 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF documents to ask questions about"
    )
    
    if st.button("🔄 Process Documents", type="primary"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                try:
                    # Create directories if they don't exist
                    os.makedirs(RAW_DIR, exist_ok=True)
                    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
                    
                    # Save uploaded files
                    all_chunks = []
                    processed_names = []
                    
                    for uploaded_file in uploaded_files:
                        # Save to raw directory
                        file_path = os.path.join(RAW_DIR, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Process the PDF
                        pages = load_pdf(file_path)
                        chunks = chunk_pages(pages)
                        all_chunks.extend(chunks)
                        processed_names.append(uploaded_file.name)
                    
                    # Embed all chunks
                    embedded_chunks = embed_chunks(all_chunks)
                    
                    # Build vector store
                    vector_store = VectorStore()
                    vector_store.add_chunks(embedded_chunks)
                    vector_store.save_to_disk()
                    
                    # Save to session state
                    st.session_state.vector_store = vector_store
                    st.session_state.processed_files = processed_names
                    
                    st.success(f"✅ Processed {len(processed_names)} document(s) with {len(all_chunks)} chunks!")
                    
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
        else:
            st.warning("Please upload at least one PDF file")
    
    st.divider()
    
    # Show processed documents
    if st.session_state.processed_files:
        st.subheader("📚 Processed Documents")
        for doc in st.session_state.processed_files:
            st.text(f"• {doc}")
    
    # Load existing index button
    st.divider()
    if st.button("📂 Load Existing Index"):
        try:
            vector_store = VectorStore()
            if vector_store.load_from_disk():
                st.session_state.vector_store = vector_store
                st.success("✅ Loaded existing index!")
            else:
                st.warning("No existing index found")
        except Exception as e:
            st.error(f"Error loading index: {str(e)}")

# Main chat area
if st.session_state.vector_store is None:
    st.info("👆 Upload and process documents using the sidebar to get started!")
else:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📎 Sources"):
                    for source in message["sources"]:
                        st.text(f"• {source['source']} (Page {source['page']}) - Relevance: {source['relevance']}")
    
    # Chat input
    if question := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(question)
        
        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Embed query and search
                    query_embedding = embed_text(question)
                    retrieved_chunks = st.session_state.vector_store.search(query_embedding)
                    
                    # Generate answer
                    result = generate_answer(question, retrieved_chunks)
                    
                    # Display answer
                    st.markdown(result['answer'])
                    
                    # Display sources
                    with st.expander("📎 Sources"):
                        for source in result['sources']:
                            st.text(f"• {source['source']} (Page {source['page']}) - Relevance: {source['relevance']}")
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result['answer'],
                        "sources": result['sources']
                    })
                    
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# Footer
st.divider()
st.caption("Built with Streamlit | RAG-powered by Mistral + FAISS | 100% Local & Private")