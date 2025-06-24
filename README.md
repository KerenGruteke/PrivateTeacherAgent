# Private Teacher Agent

An AI-powered personalized tutor designed to help students prepare for exams by generating questions, evaluating answers, providing feedback, and adapting based on progress.

---

## Features
- **Personalized Question Generation** based on user level and topic
- **Answer Evaluation** using LLM-based reasoning
- **Conceptual Feedback** to improve learning
- **Progress Tracking** with adaptive topic suggestion
- **Multi-tool Integration** (terminal, GUI cursor, retriever)

---

## Tech Stack
- **LangChain** for orchestration
- **LLMs** (OpenAI/GPT or similar)
- **Chroma/FAISS** for vector-based retrieval
- **Streamlit / CLI** for user interaction (configurable)
- **Python** (>=3.9)

---

## Setup

1. Clone the repo:
```bash
git clone https://github.com/<your_org>/private-teacher-agent.git
cd private-teacher-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add your `.env` file with API keys and configuration:
```
OPENAI_API_KEY=...
```

---

## Directory Structure
```
private-teacher-agent/
├── agents/               # Core agent logic and tool routing
├── tools/                # Terminal, cursor, retriever tool implementations
├── prompts/              # Prompt templates for each module
├── chains/               # LangChain chain definitions (questioning, feedback, etc.)
├── ui/                   # Streamlit or CLI interface
├── data/                 # Sample data, synthetic questions, etc.
├── tests/                # Unit and integration tests
├── README.md             # This file
├── requirements.txt
└── .env                  # API keys and configuration
```

---

## Usage
To launch the agent interface:
```bash
streamlit run ui/app.py
```
or run in CLI mode:
```bash
python ui/cli.py
```

---

## Contribution
We welcome contributions! Please:
- Open issues for bugs or feature suggestions
- Fork and create pull requests for code changes

---

## License
MIT License.

---

## Acknowledgments
This project was created as part of the Multi AI Agent course assignment.
Special thanks to all dataset providers and open-source maintainers.
