import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()


def get_llm():
    """Return the shared chat model. Config via .env or defaults."""
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5:14b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    )
