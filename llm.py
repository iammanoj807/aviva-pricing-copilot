import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()


def get_llm():
    """Return the chat model. Uses Groq when deployed, Ollama locally."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "groq":
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        return init_chat_model(f"groq:{model}", temperature=0)

    model = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
    return init_chat_model(f"ollama:{model}", temperature=0)
