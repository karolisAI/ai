import sys, pathlib

# Add LangChain Chatbot directory to PYTHONPATH so tests can import helpers.*
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
CHATBOT_DIR = PROJECT_ROOT / "LangChain Chatbot"

if CHATBOT_DIR.is_dir():
    sys.path.insert(0, str(CHATBOT_DIR)) 