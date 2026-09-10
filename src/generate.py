"""
generate.py
Handles: building the grounded prompt and calling the LLM (Groq - free & fast).
"""

import os
from groq import Groq
from typing import List, Dict

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context.
Rules:
- If the answer is not in the context, say "I couldn't find this in the provided documents."
- Always cite which source/chunk you used, like [Source: filename.pdf].
- Be concise and accurate. Do not make up information outside the context.
"""


def build_context(chunks: List[Dict]) -> str:
    """Format retrieved chunks into a context block with source labels."""
    parts = []
    for c in chunks:
        parts.append(f"[Source: {c['source']}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, chunks: List[Dict], model: str = "openai/gpt-oss-20b") -> str:
    """Send retrieved context + query to Groq LLM and return the grounded answer."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return (
            "⚠️ GROQ_API_KEY not set. Add it to your .env file. "
            "Get a free key at https://console.groq.com\n\n"
            f"(Here's what would've been sent as context, for debugging:)\n{build_context(chunks)[:500]}..."
        )

    client = Groq(api_key=api_key)
    context = build_context(chunks)

    user_prompt = f"""Context:
{context}

Question: {query}

Answer using only the context above, and cite sources."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return response.choices[0].message.content
