# server.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import anthropic
import openai
import google.generativeai as genai
import requests
from fastmcp import FastMCP

app = FastAPI()
mcp = FastMCP("panda")  # Initialize MCP with namespace "panda"

# Set API keys from environment variables
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLAMA_API_URL = os.getenv("LLAMA_API_URL", "http://localhost:11434/api/generate")

openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)


class QueryRequest(BaseModel):
    """Schema for LLM query request."""
    prompt: str
    model: str  # "anthropic", "openai", "llama", or "gemini"


@app.post("/query")
async def query_llm(request: QueryRequest):
    """Query the specified LLM with the provided prompt.

    Args:
        request (QueryRequest): Request object containing prompt and model selection.

    Returns:
        dict: Response from the selected LLM containing generated text.

    Raises:
        HTTPException: If an invalid model is specified or if an API error occurs.
    """
    prompt = request.prompt
    model = request.model.lower()

    if model == "anthropic":
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        completion = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = completion.content[0].text

    elif model == "openai":
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256
        )
        response_text = completion.choices[0].message.content.strip()

    elif model == "llama":
        llama_payload = {"model": "llama3", "prompt": prompt, "stream": False}
        llama_response = requests.post(LLAMA_API_URL, json=llama_payload)
        if llama_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Llama API error.")
        response_text = llama_response.json().get("response", "").strip()

    elif model == "gemini":
        gemini_model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()

    else:
        raise HTTPException(status_code=400, detail="Invalid model specified.")

    return {"response": response_text}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
