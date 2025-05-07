# agent.py
import requests
import sys
from fastmcp import FastMCP

mcp = FastMCP("panda")  # Only for namespace consistency

class WhoIsAgent:
    """Agent querying a FastMCP-based server for information using LLM."""

    def __init__(self, name: str, model: str = "openai"):
        self.name = name
        self.model = model
        self.server_url = "http://localhost:8000/query"

    def get_who_is(self) -> str:
        """Query the server using explicit HTTP to invoke MCP-defined action."""
        response = requests.post(
            self.server_url,
            json={"name": self.name, "model": self.model}
        )
        if response.status_code == 200:
            return response.json().get("response")
        else:
            return f"Error: {response.text}"

def main():
    """Entry point."""
    if len(sys.argv) != 3:
        print("Usage: python agent.py <name> <model>")
        print("Models: openai, anthropic, llama, gemini")
        sys.exit(1)

    name, model = sys.argv[1], sys.argv[2].lower()

    agent = WhoIsAgent(name, model)
    result = agent.get_who_is()
    print(f"Answer from {model.capitalize()}:\n{result}")

if __name__ == "__main__":
    main()
