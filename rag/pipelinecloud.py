# rag/pipelinecloud.py
# Cloud version using Groq API (FREE & FAST)

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
    """Generate answer using Groq API (FREE)"""
    import streamlit as st
    import requests
    
    print(f"\nGenerating answer for: '{question}'")
    
    # Get API key
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except:
        api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        raise ValueError("GROQ_API_KEY not found!")
    
    # Build context
    context = build_context(retrieved_chunks)
    prompt = create_prompt(question, context)
    
    print("\n--- Calling Groq API ---")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            st.error(f"API Error: {response.status_code} - {response.text}")
            raise Exception(f"API returned {response.status_code}")
        
        result = response.json()
        answer = result['choices'][0]['message']['content']
        
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