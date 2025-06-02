# 🏋️ AI Bodybuilding Coach

An intelligent chatbot powered by LangChain that provides personalized fitness advice, workout plans, and nutrition guidance. The application combines the power of large language models with a curated knowledge base of fitness resources.

## 🌟 Features

### Core Functionality
- **Interactive Chat Interface**: Ask any fitness-related questions in natural language
- **Special Commands**:
  - `workout plan`: Get personalized workout routines
  - `calculate calories`: Calculate daily calorie needs
  - `supplements`: Get supplement recommendations
- **Real-time Knowledge Base**: Access to latest research and guidelines from authoritative sources
- **Source Citations**: All responses include references to scientific sources

### User Experience
- **Interactive Tour**: Step-by-step guide through application features
- **Conversation History**: View and export your chat history
- **RAG Process Visualization**: See how the system retrieves and processes information
- **Responsive Design**: Clean and intuitive user interface

## 🛠️ Project Structure

```
LangChain Chatbot/
├── data/
│   ├── Nippard_Hypertrophy/     # PDF resources
│   └── extraction.py            # Data extraction utilities
├── helpers/
│   ├── baseretriever.py         # Base retriever implementation
│   ├── functions.py             # Core functionality
│   ├── monitoring.py            # Query monitoring and logging
│   ├── rag.py                   # RAG implementation
│   └── security.py              # Input validation and safety checks
├── main.py                      # Main application
├── requirements.txt             # Project dependencies
└── README.md                    # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd LangChain-Chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

### Running the Application

1. Start the Streamlit application:
```bash
streamlit run main.py
```

2. Open your browser and navigate to the provided local URL (typically http://localhost:8501)

## 📚 Knowledge Base

The application uses a combination of:
- PDF resources from Jeff Nippard's Hypertrophy Guide
- Real-time data from authoritative sources:
  - American College of Sports Medicine (ACSM)
  - National Strength and Conditioning Association (NSCA)
  - World Health Organization (WHO)
  - USDA Food Data Central
  - EatRight.org

## 🔒 Security Features

- Input validation for health metrics
- Age verification (13+ requirement)
- Exercise safety checks
- Supplement safety warnings
- Health metric validation

## 📊 Monitoring and Logging

- Query logging
- Input validation
- Rate limiting
- Security event logging

## 📝 Export Options

- JSON export of conversation history
- Source citations with relevance scores
- PDF and URL references

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Jeff Nippard for the Hypertrophy Guide
- All referenced scientific organizations and resources
- The LangChain and Streamlit communities
