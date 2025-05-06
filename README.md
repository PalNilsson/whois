# whois
A minimal example of an MCP agent and server that can be used to ask an AI about who someone is.

## Installation
Install dependencies with:
```
pip install -r requirements.txt
```

## Environment Variables
Ensure these keys are set in your environment for secure API access:
```
export ANTHROPIC_API_KEY='your_anthropic_api_key'
export OPENAI_API_KEY='your_openai_api_key'
export GEMINI_API_KEY='your_gemini_api_key'
export LLAMA_API_URL='http://localhost:11434/api/generate'  # For Ollama Llama3 model
```

## Usage
1. Run the Server:
```
uvicorn server:app --reload
```

2. Run the Agent (example queries):
```
python agent.py "Albert Einstein" openai
python agent.py "Richard Feynman" anthropic
python agent.py "Nikola Tesla" llama
python agent.py "Stephen Hawking" gemini
```


