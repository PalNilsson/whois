# server.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastmcp import FastMCP
import anthropic
import openai
import google.generativeai as genai
import requests

app = FastAPI()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLAMA_API_URL = os.getenv("LLAMA_API_URL", "http://localhost:11434/api/generate")

openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

class QueryRequest(BaseModel):
    name: str
    model: str

# Explicit MCP subclass
class PandaMCP(FastMCP):
    def who_is(self, name: str, model: str) -> str:
        prompt = f"Who is {name}?"

        if model == "anthropic":
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            completion = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}]
            )
            return completion.content[0].text.strip()

        elif model == "openai":
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256
            )
            return completion.choices[0].message.content.strip()

        elif model == "llama":
            llama_payload = {"model": "llama3", "prompt": prompt, "stream": False}
            llama_response = requests.post(LLAMA_API_URL, json=llama_payload)
            llama_response.raise_for_status()
            return llama_response.json().get("response", "").strip()

        elif model == "gemini":
            gemini_model = genai.GenerativeModel('models/gemini-1.5-flash')
            response = gemini_model.generate_content(prompt)
            return response.text.strip()

        raise ValueError(f"Unsupported model '{model}'.")

mcp = PandaMCP("panda")

@app.post("/query")
async def query_llm(request: QueryRequest):
    response = mcp.who_is(name=request.name, model=request.model)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
