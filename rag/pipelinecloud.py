# Cloud version using Anthropic Claude API instead of local Ollama
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
    """Generate answer using Claude API via direct HTTP request"""
    import streamlit as st
    import requests
    
    print(f"\nGenerating answer for: '{question}'")
    
    # Get API key
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except:
        api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found!")
    
    # Build context
    context = build_context(retrieved_chunks)
    prompt = create_prompt(question, context)
    
    print("\n--- Calling Claude API ---")
    
    # Direct HTTP request to Anthropic API
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        # Print error details for debugging
        if response.status_code != 200:
            st.error(f"API Error: {response.status_code} - {response.text}")
            raise Exception(f"API returned {response.status_code}")
        
        result = response.json()
        answer = result['content'][0]['text']
        
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