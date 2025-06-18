import os  # noqa: E402
import sys  # noqa: E402
import pytest

# Add LangChain Chatbot directory to path
CURRENT_DIR = os.path.dirname(__file__)
CHATBOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', 'LangChain Chatbot'))
if CHATBOT_DIR not in sys.path:
    sys.path.append(CHATBOT_DIR)

from helpers.rag import get_rag_chain  # noqa: E402

PDF_DIR = os.path.join('.', 'data', 'Nippard_Hypertrophy')

@pytest.mark.integration
def test_rag_chain_basic_invoke():
    """Ensure the RAG chain can be created and invoked without raising."""
    chain = get_rag_chain(PDF_DIR, urls=[])
    result = chain.invoke({"question": "What is hypertrophy?", "chat_history": []})
    assert "answer" in result or "result" in result
    text = result.get("answer") or result.get("result")
    assert isinstance(text, str) 