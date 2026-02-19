# frontend/app_cloud.py
# Cloud deployment version - Mobile optimized

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

# Import cloud-specific components
from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_pages
from embeddings.embeddings import embed_chunks, embed_text
from retrieval.vector_store import VectorStore
from rag.pipelinecloud import generate_answer  # Use cloud version
from config import RAW_DIR, VECTOR_STORE_DIR


# Page config
st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="📄",
    layout="wide"
)

# Check for API key
if not os.getenv("GROQ_API_KEY"):
    try:
        _ = st.secrets["GROQ_API_KEY"]
    except:
        st.error("⚠️ GROQ_API_KEY not found! Please add it to Streamlit secrets.")
        st.stop()

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
    st.header("📁 Upload Document")
    
    # Single file upload for better mobile support
    uploaded_file = st.file_uploader(
        "Choose ONE PDF file",
        type=['pdf'],
        accept_multiple_files=False,
        help="Upload a PDF document (max 10MB)"
    )
    
    if uploaded_file:
        st.success(f"✅ File ready: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
        
        if st.button("🔄 Process Document", type="primary"):
            with st.spinner("Processing document..."):
                try:
                    # Create directories
                    os.makedirs(RAW_DIR, exist_ok=True)
                    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
                    
                    st.info(f"Processing: {uploaded_file.name}")
                    
                    # Check file size
                    if uploaded_file.size > 10 * 1024 * 1024:
                        st.error(f"❌ File too large (max 10MB)")
                        st.stop()
                    
                    # Save file
                    file_path = os.path.join(RAW_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.success(f"✅ Saved")
                    
                    # Process
                    st.info("Extracting text...")
                    pages = load_pdf(file_path)
                    st.success(f"✅ Extracted {len(pages)} pages")
                    
                    st.info("Chunking...")
                    chunks = chunk_pages(pages)
                    st.success(f"✅ Created {len(chunks)} chunks")
                    
                    # Embed
                    st.info("Generating embeddings...")
                    embedded_chunks = embed_chunks(chunks)
                    st.success("✅ Embeddings done")
                    
                    # Build index
                    st.info("Building index...")
                    vector_store = VectorStore()
                    vector_store.add_chunks(embedded_chunks)
                    vector_store.save_to_disk()
                    
                    st.session_state.vector_store = vector_store
                    st.session_state.processed_files = [uploaded_file.name]
                    
                    st.success(f"🎉 Ready to ask questions!")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.error(f"Details: {traceback.format_exc()}")
    else:
        st.info("👆 Select a PDF file above")
    
    st.divider()
    
    if st.session_state.processed_files:
        st.subheader("📚 Processed Document")
        for doc in st.session_state.processed_files:
            st.text(f"• {doc}")
    
    st.divider()
    if st.button("📂 Load Existing Index"):
        try:
            vector_store = VectorStore()
            if vector_store.load_from_disk():
                st.session_state.vector_store = vector_store
                st.success("✅ Loaded!")
            else:
                st.warning("No index found")
        except Exception as e:
            st.error(f"Error: {str(e)}")


# Main chat area
if st.session_state.vector_store is None:
    st.info("👆 Upload and process a document to get started!")
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
    if question := st.chat_input("Ask a question about your document..."):
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })
        
        with st.chat_message("user"):
            st.markdown(question)
        
        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    query_embedding = embed_text(question)
                    retrieved_chunks = st.session_state.vector_store.search(query_embedding)
                    result = generate_answer(question, retrieved_chunks)
                    
                    st.markdown(result['answer'])
                    
                    with st.expander("📎 Sources"):
                        for source in result['sources']:
                            st.text(f"• {source['source']} (Page {source['page']}) - Relevance: {source['relevance']}")
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result['answer'],
                        "sources": result['sources']
                    })
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


st.divider()
st.caption("Built with Streamlit | Powered by Groq + FAISS | 100% Free")