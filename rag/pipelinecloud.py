# rag/pipelinecloud.py
# Cloud version using Hugging Face Inference API (FREE)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TOP_K_RESULTS


def build_context(retrieved_chunks: list) -> str:
    """Format retrieved chunks into context string"""
    context_parts = []
    
    for i, (chunk, distance) in enumerate(retrieved_chunks, 1):
        source_info = f"[Source: {chunk['source']}, Page {chunk['page_number']}]"
        chunk_text = chunk['text']
        context_parts.append(f"Chunk {i} {source_info}:\n{chunk_text}\n")
    
    return "\n".join(context_parts)


def create_prompt(question: str, context: str) -> str:
    """Create prompt for the LLM"""
    prompt = f"""You are a helpful AI assistant that answers questions based on the provided document context.

IMPORTANT INSTRUCTIONS:
- Only use information from the context below to answer the question
- If the answer is not in the context, say "I cannot find this information in the provided documents"
- Always cite which source and page number you got the information from
- Be concise and accurate

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    
    return prompt


def generate_answer(question: str, retrieved_chunks: list) -> dict:
    """Generate answer using Hugging Face Inference API (FREE)"""
    import streamlit as st
    import requests
    
    print(f"\nGenerating answer for: '{question}'")
    
    # Get API token
    try:
        api_token = st.secrets["HUGGINGFACE_API_TOKEN"]
    except:
        api_token = os.getenv("HUGGINGFACE_API_TOKEN")
    
    if not api_token:
        raise ValueError("HUGGINGFACE_API_TOKEN not found!")
    
    # Build context
    context = build_context(retrieved_chunks)
    prompt = create_prompt(question, context)
    
    print("\n--- Calling Hugging Face API ---")
    
    # Use Mistral-7B-Instruct via Hugging Face
    API_URL = "https://router.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.2,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code != 200:
            st.error(f"API Error: {response.status_code} - {response.text}")
            raise Exception(f"API returned {response.status_code}")
        
        result = response.json()
        
        # Handle different response formats
        if isinstance(result, list) and len(result) > 0:
            answer = result[0].get('generated_text', str(result))
        else:
            answer = str(result)
        
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        raise
    
    # Extract sources
    sources = []
    for chunk, distance in retrieved_chunks:
        source_info = {
            'source': chunk['source'],
            'page': chunk['page_number'],
            'relevance': round(1 / (1 + distance), 2)
        }
        if source_info not in sources:
            sources.append(source_info)
    
    return {
        'answer': answer,
        'sources': sources
    }